# Strategic Proposal: Post-Sale Credit Risk Scoring Model

## 1. Business Problem

Customer default directly impacts SunCulture's bottom line through write-offs, reduced liquidity, and increased collection operational costs. Without a systematic way to identify high-risk accounts early, collections efforts are reactive rather than preventive. Improving collections rates by even 5% would yield significant financial and operational benefits.

## 2. Proposed Solution

A predictive **Post-Sale Credit Risk Scoring Model** that assigns a risk score to each active loan account weekly. Scores feed into tiered collection strategies:

- **Low risk** → automated reminders (SMS/USSD)
- **Medium risk** → proactive check-in calls
- **High risk** → priority outreach + payment plan offers

This shifts collections from reactive to proactive, enabling early intervention before accounts become severely delinquent.

## 3. Data & Features (New Sources)

1. **Mobile Money Transaction History (M-Pesa/Airtel):** Frequency, volume, and consistency of incoming payments — a strong signal of repayment capacity.
2. **Weather / Agricultural Calendar Data:** For asset-backed loans (solar irrigation), seasonality affects cash flow. Historical rainfall vs. repayment correlation.
3. **Customer Support Interaction Logs:** Number of support tickets, complaints, or queries about payments — early behavioural red flags.
4. **Credit Bureau Data (TransUnion/Metropol):** External credit history, active loan count, and default history across other lenders.
5. **Device Telemetry (IoT):** Usage data from solar assets — reduced usage may signal financial distress before payment is missed.

## 4. Methodology

- **Algorithm:** Gradient Boosted Trees (XGBoost/LightGBM) for their handling of mixed data types and missing values. Calibrated probabilities for direct risk scoring.
- **Workflow integration:** A weekly Airflow DAG scores active accounts. Scores are written to a `collections_priority` table consumed by the collections CRM.
- **Human-in-the-loop:** Accounts above the high-risk threshold are reviewed weekly by the collections team before action is taken.

## 5. Success Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| On-time repayment rate | % of accounts paying within 30 days of due date | +10% vs. baseline |
| Default rate | % of accounts written off | -20% vs. baseline |
| Collections efficiency | Cost per dollar recovered | -15% vs. baseline |
| Early detection rate | % of defaults flagged 30+ days before write-off | > 70% |

## 6. Immediate Data-Driven Recommendations (Existing Dataset)

These recommendations use **only the provided dataset** and require no new data sources. Each is specific, measurable, and practical to implement within 4–6 weeks.

### Recommendation 1: Tiered Regional Collection Playbooks

| Element | Detail |
|---------|--------|
| **Finding** | Kenya (11.4% arrears) and Uganda (12.6%) outperform CIV (12.7%) in arrears control. However, Kenya has the highest write-off rate (14.1%) despite the lowest arrears — suggesting accounts may go directly from current to bad debt without an arrears stage. CIV has the highest repossession rate (13.2%). |
| **Action** | (a) Investigate Kenya's direct-to-bad-debt pattern — are accounts skipping arrears, or is the status classification inconsistent? (b) Deploy CIV-specific early intervention (day-15 SMS vs. current day-30) to reduce repossession risk. (c) Codify Uganda's balanced collection approach as the baseline SOP. |
| **Measurable target** | Reduce CIV repossession rate from 13.2% to ~12% within 3 months. Investigate Kenya write-off pathway and establish day-21 intervention trigger for accounts showing early warning signs. |
| **Practical implementation** | Update the Airflow DAG to export a daily `collections_queue` table sorted by region + risk tier. Collections CRM pulls from this table. No new infrastructure needed. |

### Recommendation 2: Risk-Adjusted Minimum Deposits by Product

| Element | Detail |
|---------|--------|
| **Finding** | Three products drive disproportionately high write-off rates: Water Pump Kit (15.3%), Smartphone (14.9%), and Tablet Device (14.7%). In contrast, the safest product (Premium Solar Fan) writes off at 12.4% — a narrow 2.9pp range across all products, suggesting risk is mostly customer-driven, not product-driven. PAYG accounts write off at 14.2% vs. 13.0% for CASH (1.09× multiple). |
| **Action** | Increase minimum deposit from 10% to 20% for Water Pump, Smartphone, and Tablet accounts. For PAYG accounts on these products, require 25% minimum deposit or a co-signer. |
| **Measurable target** | Reduce write-off rate on these three products from ~15% to ~12% within 6 months. Expected impact: ~40 fewer write-offs annually, saving ~$80K–$120K in losses (assuming $2K–$3K average product value). |
| **Practical implementation** | Update deposit validation logic in the account creation workflow (single SQL config table or application config change). No model deployment required. Monitor impact via existing status tracking. |

### Recommendation 3: PAYG Automated Disconnection Tier

| Element | Detail |
|---------|--------|
| **Finding** | PAYG accounts have 13.3% arrears rate vs. 11.2% for CASH (1.19× higher). They also write off at 14.2% vs. 13.0%. However, PAYG's built-in remote disconnection capability enables a uniquely effective collection lever. |
| **Action** | Implement a graduated PAYG disconnection schedule: (1) Day-7 missed payment → SMS warning. (2) Day-14 → system reduces power output by 50% (soft lock). (3) Day-21 → full disconnect. (4) Day-30 → escalate to field agent. Reconnect automatically upon payment + $5 reconnection fee. |
| **Measurable target** | Reduce PAYG arrears from 13.3% to 10.5% within 4 months — a ~21% relative reduction. Target 60% of PAYG arrears accounts resolved before day-21 disconnect. |
| **Practical implementation** | IoT team configures the device management platform with the graduated schedule. Collections team monitors disconnect/reconnect metrics via existing dashboard. No model or data pipeline changes required. |

### Recommendation 4: Gender-Smart Collection Messaging

| Element | Detail |
|---------|--------|
| **Finding** | Male customers default at 40.1% vs. 36.0% for females — a 4.1pp gap. However, gender may be confounded by region or product: male and female customers may systematically buy different products or be concentrated in different regions. |
| **Action** | Before implementing gender-based policies: (a) Run a logistic regression with interaction terms to test whether gender is independently predictive or confounded. (b) If independently predictive, pilot gender-parameterized SMS templates — female customers: flexibility and support; male customers: urgency and consequences. (c) If confounded, address the root causes (product mix, regional practices) rather than targeting gender directly. |
| **Measurable target** | Determine within 4 weeks whether gender is independently predictive. If yes, pilot A/B test on 500 accounts per arm and target closing the gap from 3.8pp to <1pp within 6 months. |
| **Practical implementation** | Update SMS/USSD templates in the collections CRM with gender-parameterized messaging. A/B test results logged to Airflow metadata DB for analysis. |

## 7. Next Steps (POC)

1. **Data exploration:** Access and profile the 5 new data sources; negotiate data-sharing agreements.
2. **Feature engineering:** Create temporal features (rolling averages, trend indicators) from transaction history.
3. **Model development:** Train baseline model using existing internal data; iteratively add new features.
4. **Validation:** Walk-forward time-series cross-validation to simulate production conditions.
5. **Stakeholder engagement:** Present POC results to Credit Collections and Product teams; refine tier strategy.
