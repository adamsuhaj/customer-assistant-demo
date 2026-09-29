# Customer Facing Assistant Demo Dataset

All customer, contact, instrument serial, order, shipment, tracking, and service-event records in these CSVs are fictional and created for a software demonstration. IDs beginning with DEMO- are synthetic. Email addresses use reserved example domains.

The product family name and troubleshooting reference point to Agilent's public 1260 Infinity II pump user manual. The knowledge cards are short demo summaries, not copied manual text or a replacement for the manufacturer's instructions. Keep the assistant within authorized operator guidance and escalate suspected leaks, component faults, or persistent errors to qualified service support.

Public source:
https://www.agilent.com/cs/library/usermanuals/public/G7111AUser.pdf
Relevant topic: Error Information > Pressure Below Lower Limit (printed page 108).

## Files and joins
- customers.customer_id joins to instruments.customer_id, orders.customer_id, and service_history.customer_id.
- instruments.instrument_id joins to orders.instrument_id and service_history.instrument_id.
- troubleshooting_articles contains public guidance and its source locator.
- evaluation_cases includes one intentional wrong-answer fixture. EVAL-005 contains a deliberately wrong candidate_override_text. It must fail grounding and block promotion; never show it to a customer.

## Intentional inconsistent source records

- Alice's extra order DEMO-ORD-1009 is in transit, shipped on 2026-09-26, with estimated delivery on 2026-10-02.
- Her closed service event DEMO-SVC-1009 claims that the same shipment, DEMO-TRACK-1009, was received on 2026-09-23 and its check valve kit installed on 2026-09-24, before the shipping date.
- These two records deliberately demonstrate contradictory source data. They do not create a new evaluation case: the existing grounding checks do not compare order records with service records, so this example alone does not produce an automatic grounding failure or promotion verdict.
- To compare them in Streamlit, select Alice and ask separately: "Where is order DEMO-ORD-1009?" and "Show my service history for DEMO-INS-1001."

## Demo story
- Select Alice, then ask about DEMO-ORD-1007 or DEMO-INS-1001.
- Select Bob and try to read Alice's order. The tool must deny before returning the row to the model.
- Ask about the pressure alert to retrieve DEMO-KB-001.
- Run EVAL-005 to show that a plausible but wrong “delivered” claim does not pass the grounding gate.
