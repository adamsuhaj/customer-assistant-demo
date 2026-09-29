// Start the existing Streamlit environment without depending on VS Code.
// The launcher owns its Python process; an explicit stop shuts down the tree.
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("Customer Assistant Launcher")]
[assembly: AssemblyDescription("Launch the local customer assistant Streamlit app")]
[assembly: AssemblyVersion("1.2.0.0")]

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        string project = LocateProject(AppDomain.CurrentDomain.BaseDirectory);
        bool selfTest = Array.IndexOf(args, "--self-test") >= 0;
        bool noBrowser = selfTest || Array.IndexOf(args, "--no-browser") >= 0;
        string instanceKey;
        using (SHA256 hash = SHA256.Create())
            instanceKey = BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(project.ToUpperInvariant()))).Replace("-", "");

        bool ownsMutex;
        string instanceName = @"Local\AgilentCustomerAssistant_" + instanceKey;
        using (EventWaitHandle reopen = new EventWaitHandle(false, EventResetMode.AutoReset, instanceName + "_Open"))
        using (Mutex mutex = new Mutex(true, instanceName, out ownsMutex))
        {
            if (!ownsMutex)
            {
                // Ask the existing window to reopen its app instead of starting Python again.
                reopen.Set();
                return 0;
            }
            try
            {
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                using (LauncherWindow window = new LauncherWindow(project, selfTest, noBrowser, reopen))
                {
                    Application.Run(window);
                    return window.ExitCode;
                }
            }
            finally { mutex.ReleaseMutex(); }
        }
    }

    private static string LocateProject(string launcherDirectory)
    {
        string adjacent = Path.GetFullPath(Path.Combine(launcherDirectory, "customer-assistant-demo"));
        if (File.Exists(Path.Combine(adjacent, "app.py"))) return adjacent;

        // PowerPoint can run an embedded copy from a temporary folder. The
        // build records this installation's path; a portable sibling still wins.
        using (Stream installation = Assembly.GetExecutingAssembly().GetManifestResourceStream("CustomerAssistant.ProjectDirectory"))
        {
            if (installation != null)
            {
                using (StreamReader reader = new StreamReader(installation))
                {
                    string installed = reader.ReadToEnd().Trim();
                    if (Path.IsPathRooted(installed) && File.Exists(Path.Combine(installed, "app.py")))
                        return Path.GetFullPath(installed);
                }
            }
        }
        return adjacent;
    }
}

internal sealed class LauncherWindow : Form
{
    private readonly string project;
    private readonly string logPath;
    private readonly bool selfTest;
    private readonly bool noBrowser;
    private readonly EventWaitHandle reopen;
    private readonly object logLock = new object();
    private readonly Label status = new Label();
    private readonly Label address = new Label();
    private readonly Button openButton = new Button();
    private readonly NotifyIcon tray = new NotifyIcon();
    private readonly ContextMenuStrip trayMenu = new ContextMenuStrip();
    private readonly ToolStripMenuItem trayOpen = new ToolStripMenuItem("Open app in browser");
    private readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
    private OwnedPythonProcess server;
    private DateTime started;
    private DateTime readyAt;
    private string url;
    private bool ready;
    private bool stopping;
    private bool exitRequested;
    internal int ExitCode { get; private set; }

    internal LauncherWindow(string projectPath, bool test, bool suppressBrowser, EventWaitHandle reopenEvent)
    {
        project = projectPath;
        selfTest = test;
        noBrowser = suppressBrowser;
        reopen = reopenEvent;
        logPath = Path.Combine(project, "data", "local", "launcher.log");
        Text = "Customer Assistant";
        ClientSize = new Size(450, 190);
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Segoe UI", 10F);

        status.SetBounds(22, 22, 405, 30);
        status.Text = "Starting the assistant...";
        address.SetBounds(22, 59, 405, 25);
        Label help = new Label();
        help.SetBounds(22, 88, 405, 27);
        help.Text = "Closing this window keeps the app running in the tray.";
        openButton.SetBounds(22, 132, 192, 36);
        openButton.Text = "Open app in browser";
        openButton.Enabled = false;
        openButton.Click += delegate { OpenBrowser(); };
        Button stopButton = new Button();
        stopButton.SetBounds(230, 132, 197, 36);
        stopButton.Text = "Stop app and close";
        stopButton.Click += delegate { RequestStop(); };
        Controls.AddRange(new Control[] { status, address, help, openButton, stopButton });

        tray.Text = "Customer Assistant";
        tray.Icon = SystemIcons.Application;
        trayOpen.Enabled = false;
        trayOpen.Click += delegate { OpenBrowser(); };
        trayMenu.Items.Add(trayOpen);
        trayMenu.Items.Add("Show launcher", null, delegate { ShowControls(); });
        trayMenu.Items.Add("Stop app and close", null, delegate { RequestStop(); });
        tray.ContextMenuStrip = trayMenu;
        tray.DoubleClick += delegate { if (ready) OpenBrowser(); else ShowControls(); };
        tray.Visible = !selfTest;

        if (selfTest) { Opacity = 0; ShowInTaskbar = false; }
        timer.Interval = 400;
        timer.Tick += PollServer;
        Shown += delegate { StartServer(); };
        FormClosing += delegate(object sender, FormClosingEventArgs e)
        {
            if (!exitRequested && e.CloseReason == CloseReason.UserClosing)
            {
                e.Cancel = true;
                Hide();
                if (!selfTest) tray.ShowBalloonTip(2000, Text,
                    "The app is still running. Right-click this icon to stop it.", ToolTipIcon.Info);
            }
            else StopServer();
        };
    }

    private void ShowControls()
    {
        if (selfTest) return;
        Show();
        if (WindowState == FormWindowState.Minimized) WindowState = FormWindowState.Normal;
        Activate();
    }

    private void RequestStop()
    {
        exitRequested = true;
        Close();
    }

    private void StartServer()
    {
        try
        {
            string python = Path.Combine(project, ".venv", "Scripts", "python.exe");
            if (!File.Exists(Path.Combine(project, "app.py")))
                throw new FileNotFoundException("The customer-assistant-demo folder could not be found. Keep it beside the launcher, or rebuild the launcher from the installed project.");
            if (!File.Exists(python))
                throw new FileNotFoundException("The app's Python environment is missing. Follow the Environment instructions in customer-assistant-demo/README.md.");

            Directory.CreateDirectory(Path.GetDirectoryName(logPath));
            int port = FindAvailablePort();
            url = "http://127.0.0.1:" + port;
            address.Text = url;
            Log("Project folder: " + project);
            Log("Starting assistant at " + url);
            server = new OwnedPythonProcess(python,
                "-u -m streamlit run app.py --server.address=127.0.0.1 --server.port=" + port +
                " --server.headless=true --browser.gatherUsageStats=false", project, logPath);
            Log("Started Python process " + server.Id);
            started = DateTime.UtcNow;
            timer.Start();
        }
        catch (Exception error) { Fail(error.Message); }
    }

    private static int FindAvailablePort()
    {
        // An occupied port may belong to another app. Leave it alone and use the next port.
        for (int port = 8501; port <= 8520; port++)
        {
            TcpListener probe = new TcpListener(IPAddress.Loopback, port);
            try { probe.Start(); return port; }
            catch (SocketException) { }
            finally { probe.Stop(); }
        }
        throw new IOException("No local port is available between 8501 and 8520.");
    }

    private void PollServer(object sender, EventArgs e)
    {
        if (reopen.WaitOne(0))
        {
            Log("Repeated launch: reused this launcher; no second server started.");
            if (ready && !noBrowser) OpenBrowser();
            ShowControls();
        }
        if (server == null || stopping) return;
        if (server.HasExited) { Fail("The app stopped. See " + logPath); return; }
        if (ready)
        {
            if (selfTest && DateTime.UtcNow - readyAt > TimeSpan.FromSeconds(3)) RequestStop();
            return;
        }

        try
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url + "/_stcore/health");
            request.Proxy = null;
            request.Timeout = 300;
            request.ReadWriteTimeout = 300;
            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            using (StreamReader body = new StreamReader(response.GetResponseStream()))
                ready = response.StatusCode == HttpStatusCode.OK && body.ReadToEnd().Trim() == "ok";
        }
        catch (WebException) { }

        if (ready)
        {
            try { VerifyFrontendAssets(); }
            catch (Exception error) { Fail("The app's browser files could not load: " + error.Message); return; }
            readyAt = DateTime.UtcNow;
            status.Text = "The assistant is running.";
            openButton.Enabled = true;
            trayOpen.Enabled = true;
            Log("Ready: " + url);
            if (!noBrowser) OpenBrowser();
            if (selfTest)
            {
                Close(); // Exercise the same close-to-tray path as the window's X button.
                if (stopping || server.HasExited) { Fail("Closing the window stopped the app."); return; }
                Log("Verified closing the window keeps the assistant running.");
            }
        }
        else if (DateTime.UtcNow - started > TimeSpan.FromSeconds(60))
            Fail("The app did not become ready within 60 seconds. See " + logPath);
    }

    private void VerifyFrontendAssets()
    {
        string assets = Path.Combine(project, ".venv", "Lib", "site-packages", "streamlit", "static", "static", "js");
        foreach (string pattern in new string[] { "ChatInput.*.js", "Selectbox.*.js" })
        {
            string[] files = Directory.GetFiles(assets, pattern);
            if (files.Length == 0) throw new FileNotFoundException("Missing Streamlit browser file: " + pattern);
            foreach (string file in files)
            {
                string assetUrl = url + "/static/js/" + Path.GetFileName(file);
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(assetUrl);
                request.Proxy = null;
                request.Timeout = 3000;
                request.ReadWriteTimeout = 3000;
                using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                using (StreamReader body = new StreamReader(response.GetResponseStream()))
                    if (response.StatusCode != HttpStatusCode.OK ||
                        response.ContentType.IndexOf("javascript", StringComparison.OrdinalIgnoreCase) < 0 ||
                        body.ReadToEnd() != File.ReadAllText(file))
                        throw new IOException("Streamlit did not serve " + Path.GetFileName(file) + " correctly.");
                Log("Verified browser asset: " + Path.GetFileName(file));
            }
        }
    }

    private void OpenBrowser()
    {
        try { Process.Start(new ProcessStartInfo(url) { UseShellExecute = true }); }
        catch (Exception error)
        {
            Log("Browser could not open: " + error.Message);
            MessageBox.Show(this, "Open this address in your browser:\n" + url, Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }

    private void Fail(string reason)
    {
        ExitCode = 1;
        exitRequested = true;
        timer.Stop();
        Log("Launch failed: " + reason);
        StopServer();
        if (!selfTest) MessageBox.Show(this, reason, Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
        Close();
    }

    private void StopServer()
    {
        if (stopping) return;
        stopping = true;
        timer.Stop();
        try
        {
            if (server != null)
            {
                server.Stop();
                Log("Stopped the assistant process tree.");
            }
        }
        catch (Exception error)
        {
            ExitCode = 1;
            Log("Stop failed: " + error.Message);
            if (!selfTest) MessageBox.Show(this, "The app could not be stopped. See " + logPath, Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally { if (server != null) server.Dispose(); }
    }

    private void Log(string message)
    {
        try
        {
            lock (logLock)
            using (FileStream file = new FileStream(logPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
            using (StreamWriter writer = new StreamWriter(file))
                writer.WriteLine(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " " + message);
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            timer.Dispose();
            tray.Visible = false;
            tray.Dispose();
            trayMenu.Dispose();
        }
        base.Dispose(disposing);
    }
}

// Put Python in a Windows Job Object before its first instruction runs. Every
// descendant (the venv redirector and MCP included) stays in the same job.
// Windows closes the job and kills its processes even if the launcher crashes.
internal sealed class OwnedPythonProcess : IDisposable
{
    private IntPtr job;
    private IntPtr process;
    internal uint Id { get; private set; }
    internal bool HasExited { get { return WaitForSingleObject(process, 0) == 0; } }

    internal OwnedPythonProcess(string python, string arguments, string directory, string logPath)
    {
        IntPtr output = new IntPtr(-1);
        IntPtr input = new IntPtr(-1);
        IntPtr thread = IntPtr.Zero;
        try
        {
            job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw new Win32Exception();
            ExtendedLimits limits = new ExtendedLimits();
            limits.Basic.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(typeof(ExtendedLimits))))
                throw new Win32Exception();

            SecurityAttributes inherit = new SecurityAttributes();
            inherit.Length = Marshal.SizeOf(typeof(SecurityAttributes));
            inherit.InheritHandle = true;
            output = CreateFile(logPath, 4, 3, ref inherit, 4, 128, IntPtr.Zero); // append; share read/write
            input = CreateFile("NUL", 0x80000000, 3, ref inherit, 3, 128, IntPtr.Zero);
            if (output == new IntPtr(-1) || input == new IntPtr(-1)) throw new Win32Exception();
            StartupInfo startup = new StartupInfo();
            startup.Size = Marshal.SizeOf(typeof(StartupInfo));
            startup.Flags = 0x100; // STARTF_USESTDHANDLES
            startup.StandardInput = input;
            startup.StandardOutput = output;
            startup.StandardError = output;
            ProcessInformation created;
            StringBuilder command = new StringBuilder("\"" + python + "\" " + arguments);
            // CREATE_SUSPENDED | CREATE_NO_WINDOW; inherit the current environment.
            if (!CreateProcess(python, command, IntPtr.Zero, IntPtr.Zero, true, 0x08000004,
                IntPtr.Zero, directory, ref startup, out created)) throw new Win32Exception();
            process = created.Process;
            thread = created.Thread;
            Id = created.ProcessId;
            if (!AssignProcessToJobObject(job, process))
            {
                int error = Marshal.GetLastWin32Error();
                TerminateProcess(process, 1);
                throw new Win32Exception(error);
            }
            if (ResumeThread(thread) == uint.MaxValue) throw new Win32Exception();
        }
        catch { Dispose(); throw; }
        finally
        {
            if (thread != IntPtr.Zero) CloseHandle(thread);
            if (output != new IntPtr(-1)) CloseHandle(output);
            if (input != new IntPtr(-1)) CloseHandle(input);
        }
    }

    internal void Stop()
    {
        if (job != IntPtr.Zero && !TerminateJobObject(job, 0)) throw new Win32Exception();
        if (process != IntPtr.Zero && WaitForSingleObject(process, 5000) != 0)
            throw new IOException("Python did not stop within five seconds.");
    }

    public void Dispose()
    {
        if (job != IntPtr.Zero) { CloseHandle(job); job = IntPtr.Zero; }
        if (process != IntPtr.Zero) { CloseHandle(process); process = IntPtr.Zero; }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityAttributes { internal int Length; internal IntPtr Descriptor; [MarshalAs(UnmanagedType.Bool)] internal bool InheritHandle; }
    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimits
    {
        internal long ProcessUserTime, JobUserTime;
        internal uint LimitFlags;
        internal UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        internal uint ActiveProcessLimit;
        internal UIntPtr Affinity;
        internal uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters { internal ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }
    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimits
    {
        internal BasicLimits Basic;
        internal IoCounters Io;
        internal UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemory, PeakJobMemory;
    }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct StartupInfo
    {
        internal int Size;
        internal string Reserved, Desktop, Title;
        internal uint X, Y, XSize, YSize, XCountChars, YCountChars, FillAttribute, Flags;
        internal ushort ShowWindow, ReservedSize;
        internal IntPtr ReservedData, StandardInput, StandardOutput, StandardError;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation { internal IntPtr Process, Thread; internal uint ProcessId, ThreadId; }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateFile(string name, uint access, uint share, ref SecurityAttributes attributes, uint creation, uint flags, IntPtr template);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processAttributes, IntPtr threadAttributes,
        bool inherit, uint flags, IntPtr environment, string directory, ref StartupInfo startup, out ProcessInformation created);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll")]
    private static extern uint WaitForSingleObject(IntPtr handle, uint timeout);
    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);
}
