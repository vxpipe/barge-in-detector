Write natural caller phrases spoken WHILE a voice agent is still talking.
The application already distinguishes overlap from speech after the agent stops.
Backchannels acknowledge, follow, agree, react, thank, or encourage without
claiming the floor. Barge-ins stop, correct, disagree, ask, or redirect.

Follow the requested language, regional variety, script, intent and phrase form.
Use very common everyday expressions, including the simplest obvious ones.
Do not invent unusual expressions merely for variety. Do not use romanization,
transcription noise, stage directions, brackets or ellipses. Use natural casing
and punctuation; a separate cleaner normalizes the text afterwards.
Do not supply agent dialogue or depend on an unseen question. Every sample
must have the requested label. A backchannel must not contain a hidden request,
correction, disagreement, or statement that makes the agent's speech unnecessary.

Return JSON {"samples": [{"text": "...", "anchor": "...", "label": "...",
"rationale": "..."}]}. The rationale is at most eight words. The anchor is the
short common backchannel at the START of the text; for a core phrase it equals
the text. For a direct short interruption, anchor is the empty string.
For mixed-language requests the complete text must include both languages,
but the anchor itself can be in either one. Give distinct cleaned texts in a
request. Keep every complete text within one spoken sentence and 30 words.

For English-only criteria (including Indian English), do not insert Hindi or
other Indian-language words in Latin letters, such as haan, achha, or arre.
Such words belong only to an explicitly mixed-language criterion and must be
written in that Indian language’s native script. Regional English can still
use its normal English vocabulary and discourse particles.
