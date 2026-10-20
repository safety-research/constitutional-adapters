// Synthetic stand-in for G0DM0D3's src/lib/libertas.ts, read by the test suite only.
// It keeps the upstream file's layout (one HALL_OF_FAME array of object literals, the two ids the
// attack selects, the same {Z}/{QUERY} slots in the same fields) and carries invented, benign text.
// The real file is fetched at run time by sisteer.eval.attacks_godmode and is never in this repository.

export interface HallOfFameCombo {
  id: string           // unique key
  model: string        // model id
  codename: string     // display name
  description: string  // one-line description
  color: string        // badge colour (hex)
  system: string       // system prompt with {Z}/{QUERY} placeholders
  user: string         // user prompt with {Z}/{QUERY} placeholders
  fast?: boolean       // stream immediately
}

export const HALL_OF_FAME: HallOfFameCombo[] = [
  // -- 1. first stand-in: two {Z} slots in system, one {QUERY} slot in user ------------------------
  {
    id: 'grok-420',
    model: 'example/model-a',
    codename: 'GROK 4.20',
    description: 'Stand-in entry \'one\' for the reader tests',
    color: '#000001',
    system: `• Stand-in brief \u{2603}: subject [{Z}]. Reply in a \`fenced\` style \\ with an \x41 note, \
then restate {Z} plainly. Cost: \$0.`,
    user: `request: {QUERY}`,
  },

  /* -- 2. a decoy the reader must not select ------------------------------------------------------ */
  {
    id: 'decoy-entry',
    model: 'example/model-b',
    codename: 'DECOY',
    description: "A decoy with a \"double-quoted\" description 😀",
    color: '#000002',
    system: `Decoy text that also carries {QUERY}
on two lines.`,
    user: `{QUERY}`,
  },

  // -- 3. second stand-in: one {QUERY} slot in system, and the user turn is the bare slot ----------
  {
    id: 'hermes-fast',
    model: 'example/model-c',
    codename: 'GODMODE FAST',
    description: 'Stand-in entry two',
    color: '#000003',
    fast: true,
    system: `Second stand-in brief. Answer the request {QUERY} as a short list.`,
    user: `{QUERY}`,
  },
]

// The reader stops at the end of the array, so the interpolation below is never read.
export function injectQuery(text: string, query: string): string {
  return `${text}`.replaceAll('{QUERY}', query)
}
