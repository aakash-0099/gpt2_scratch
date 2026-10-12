/**
 * The token sequence, with the freshly sampled token marked.
 *
 * This is the loop made visible: the prompt fills the strip, one token is
 * appended per step, and the pipeline animates again over the longer context.
 * The newest token is distinguished by weight and colour rather than by position,
 * because at the end of a strip, position alone is hard to notice.
 */

import { useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import {
  useTraceStore,
  deriveDisplayTokens,
  selectCurrentStep,
  selectCurrentTrace,
} from '@/store/useTraceStore';

export function TokenStrip() {
  // Select only stable slices, then derive. `useTraceStore(selectCurrentStep)`
  // returns a reference into the store's own array, so it is snapshot-safe;
  // `deriveDisplayTokens` allocates, so it must sit behind `useMemo` keyed on
  // that step. Passing it to the store hook instead re-renders forever and
  // React throws — see the note on the derivation.
  const step = useTraceStore(selectCurrentStep);
  const trace = useTraceStore(selectCurrentTrace);
  const queryPosition = useTraceStore((s) => s.queryPosition);
  const selectQuery = useTraceStore((s) => s.selectQuery);
  const status = useTraceStore((s) => s.status);

  const tokens = useMemo(() => deriveDisplayTokens(useTraceStore.getState()), [step, trace]);

  return (
    <TokenStripView
      tokens={tokens}
      contextLength={trace?.input_token_ids.length ?? 0}
      queryPosition={queryPosition}
      streaming={status === 'running'}
      onSelectQuery={selectQuery}
    />
  );
}

export function TokenStripView({
  tokens,
  contextLength,
  queryPosition,
  streaming,
  onSelectQuery,
}: {
  tokens: { piece: string; isNew: boolean }[];
  /** Number of leading tokens that came from the trace's context. */
  contextLength: number;
  queryPosition: number;
  streaming: boolean;
  onSelectQuery: (position: number) => void;
}) {
  const T = contextLength;

  return (
    <section className="panel" aria-label="Token sequence">
      <header className="panel-header">
        <h2 className="panel-title">Tokens</h2>
        <span className="value text-[11px] text-slate-500">
          {T} in context
          {tokens.length > T ? ` · ${tokens.length - T} generated` : ''}
          {streaming && <span className="ml-2 text-accent-500">streaming…</span>}
        </span>
      </header>

      <div className="flex flex-wrap items-center gap-1 px-3 py-3">
        {tokens.length === 0 && (
          <p className="text-xs text-slate-500">
            run a generation, or load a trace in replay mode
          </p>
        )}

        <AnimatePresence initial={false}>
          {tokens.map((token, i) => {
            // The trace's context positions are queryable; the appended token is
            // not, because it has not been through a forward pass yet.
            const isContext = i < T;
            const selected = isContext && i === queryPosition;
            return (
              <motion.span
                key={`${i}-${token.piece}`}
                initial={{ opacity: 0, scale: 0.8, y: 4 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.22, ease: 'easeOut' }}
              >
                <button
                  type="button"
                  disabled={!isContext}
                  onClick={() => onSelectQuery(i)}
                  title={
                    isContext
                      ? `query position ${i} — click to focus attention here`
                      : 'sampled from the previous step, not yet in a forward pass'
                  }
                  className={[
                    'tnum rounded border px-1.5 py-1 text-xs transition-colors',
                    token.isNew
                      ? 'border-accent-500 bg-accent-500/20 font-semibold text-accent-300'
                      : selected
                        ? 'border-accent-500/70 bg-ink-700 text-accent-400'
                        : 'border-ink-600 bg-ink-800 text-slate-300',
                    isContext ? 'cursor-pointer hover:border-accent-500/60' : 'cursor-default',
                  ].join(' ')}
                >
                  {visible(token.piece)}
                </button>
              </motion.span>
            );
          })}
        </AnimatePresence>
      </div>

      {tokens.some((t) => t.isNew) && (
        <p className="border-t border-ink-700 px-3 py-1.5 text-[11px] text-slate-500">
          <span className="text-accent-400">highlighted</span> token was sampled from the step
          just shown; it joins the context at the next forward pass.
        </p>
      )}
    </section>
  );
}

/** Make GPT-2's whitespace markers visible without hiding the shape of a word. */
function visible(piece: string): string {
  return piece
    .replace(/\n/g, '\\n')
    .replace(/\t/g, '\\t')
    .replace(/ /g, '·');
}
