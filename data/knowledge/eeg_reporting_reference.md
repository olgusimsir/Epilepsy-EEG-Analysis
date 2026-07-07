# EEG Reporting Reference (Sample Knowledge Base)

> Educational sample corpus for the RAG demo — NOT authoritative clinical guidance.
> Replace the files in `data/knowledge/` with your own vetted epilepsy guidelines,
> EEG interpretation rules, and de-identified sample reports.

## Structure of an EEG report

A standard clinical EEG report contains a Findings section describing the observed
electrographic activity, a Clinical Interpretation (or Impression) summarizing
whether the recording is normal or abnormal, and a Recommendation for follow-up.
Reports should state the recording conditions and any limitations of the analysis.

## Describing seizure activity

When epileptiform or seizure-like activity is present, the report should note its
timing (onset and offset), duration, and the channels or regions involved. Isolated
brief discharges are described differently from sustained rhythmic ictal patterns.
The burden of abnormal activity — how much of the recording is affected — is
relevant to interpretation.

## Interpreting detection confidence and risk

Automated seizure-detection models output a confidence per analyzed epoch. A small
number of high-confidence detections forming a contiguous episode is more concerning
than scattered low-confidence epochs, which may reflect artifact. Risk should be
communicated as a level (for example low, moderate, or high) with supporting detail,
not as a definitive diagnosis.

## Decision-support framing and limitations

Automated EEG analysis is a decision-support aid, not a diagnosis. Every automated
report should recommend review by a qualified neurologist, and should be cautious
about false positives from movement, muscle, or electrode artifact. Single-channel
or short recordings limit confidence.

## Recommendation phrasing

When seizure-like activity is detected, a typical recommendation is clinical
correlation and review of the raw EEG by an epileptologist, and consideration of
longer monitoring if clinically indicated. When no abnormal activity is flagged, a
normal automated screen does not exclude epilepsy and routine clinical correlation
is advised.