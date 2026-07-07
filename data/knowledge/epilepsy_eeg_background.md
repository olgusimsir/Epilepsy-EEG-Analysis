# Epilepsy & EEG Background (Sample Knowledge Base)

> Educational sample corpus for the RAG demo — general background only, NOT
> authoritative clinical guidance or a diagnostic protocol. Replace with your own
> vetted epilepsy references and de-identified sample reports for real use.

## What an EEG measures

An electroencephalogram (EEG) records the brain's electrical activity through
electrodes placed on the scalp, following a standard placement scheme such as the
international 10-20 system. It captures voltage fluctuations over time across many
channels, and is a core tool for evaluating suspected seizures and epilepsy.

## The role of EEG in epilepsy

EEG helps characterize seizure activity and epileptiform abnormalities. A single
routine EEG can be normal even in people with epilepsy, because abnormalities may not
occur during the short recording. This is why a normal automated or manual screen does
not exclude epilepsy, and why longer or repeated monitoring is sometimes used.

## Seizures as electrographic patterns

A seizure typically appears on EEG as an evolving, rhythmic pattern that begins,
builds, and then resolves, rather than as a single isolated blip. Sustained rhythmic
activity that evolves in frequency and spreads across channels is more characteristic
of a seizure than a brief, isolated discharge. Describing how a pattern evolves over
time is central to EEG interpretation.

## Why recording-level summary matters

Individual window scores are noisy. What matters clinically is the overall picture:
how many distinct episodes occurred, how long they lasted, how confident the strongest
detection was, and what fraction of the recording was affected. A concise
recording-level summary communicates this more usefully than a raw list of flagged
windows.

## Communicating risk responsibly

Risk from an automated analysis is best expressed as a graded level — for example low,
moderate, or high — accompanied by the evidence behind it, and always framed as
decision support rather than a diagnosis. The report should state its limitations and
recommend correlation with clinical history and review of the raw EEG by a qualified
neurologist or epileptologist.

## Limitations of automated screening

Automated detection can miss atypical seizures and can be misled by artifact. Model
performance also varies from patient to patient, so a generic model's confidence
should be read with appropriate caution. Automated output supports, but does not
replace, expert review.