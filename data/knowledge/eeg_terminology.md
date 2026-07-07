# EEG Terminology Reference (Sample Knowledge Base)

> Educational sample corpus for the RAG demo — general terminology only, NOT
> authoritative clinical guidance. Replace with your own vetted references for real use.

## Epochs and windows

Automated analysis divides a continuous EEG into short segments — commonly called
epochs or windows — and scores each one. A per-window seizure probability is the
model's estimate that a given segment contains seizure-like activity. Aggregating
these window scores gives a recording-level picture.

## Ictal, interictal, and postictal

"Ictal" refers to activity occurring during a seizure. "Interictal" refers to the
period between seizures, which may still contain epileptiform discharges such as
spikes or sharp waves. "Postictal" refers to the period immediately after a seizure,
often marked by slowing. A descriptive report distinguishes ictal patterns from
isolated interictal discharges.

## Episodes, onset, and offset

An episode is a contiguous stretch of segments flagged as abnormal. Its onset is the
start time of the first flagged segment and its offset is the end time of the last.
Duration is offset minus onset. Reporting onset, offset, and duration lets a reviewer
locate the event quickly in the raw trace.

## Focal versus generalized

Seizure activity may be described as focal, involving a limited set of channels or a
region, or generalized, involving widespread channels. Automated channel-level or
region-level information helps a reader judge whether a detection is focal or diffuse.

## Artifact

Not every abnormal-looking segment is a seizure. Movement, muscle activity, eye
blinks, chewing, and loose or faulty electrodes can all produce artifact that a model
may score as suspicious. Scattered, low-confidence, single-channel detections are more
likely to reflect artifact than a true sustained rhythmic ictal pattern.

## Sensitivity and specificity

Lowering a detection threshold increases sensitivity — more true events are caught,
but more false positives appear. Raising it increases specificity — fewer false
alarms, but subtle events may be missed. The right operating point depends on whether
a missed seizure or a false alarm is the costlier error in a given setting.