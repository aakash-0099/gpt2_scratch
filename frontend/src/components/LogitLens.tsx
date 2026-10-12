/**
 * Logit lens: the prediction as it forms, layer by layer.
 *
 * At every block depth the residual stream is pushed through the model's *own*
 * final LayerNorm and output head, and the top-5 tokens at the last prompt
 * position are read out. The deepest row is not an approximation — it is the
 * model's real prediction, and the backend asserts that equality.
 *
 * What this makes visible is that the prediction is not computed once at the
 * end: early layers produce a confident but semantically wrong answer, and the
 * answer gets progressively corrected. On real 124M weights the top token
 * usually swings several times before settling, so the arrow between rows marks
 * where the top-1 changed.
 */

import { scaleLinear } from 'd3-scale';
import { motion } from 'framer-motion';

import { formatPercent } from '@/lib/format';
import { useTraceStore } from '@/store/useTraceStore';
import type { LogitLensEntry, Trace } from '@/types/trace';

export function LogitLens({ trace }: { trace: Trace | null }) {
  const active = useTraceStore((s) => s.selectedBlock);
  const selectBlock = useTraceStore((s) => s.selectBlock);
  const lens = trace?.logit_lens ?? [];

  if (!trace || lens.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-3 py-8 text-xs text-slate-500">
        no logit lens yet
      </div>
    );
  }

  const probs = lens.flatMap((entry) => entry.top_probs);
  const x = scaleLinear().domain([0, Math.max(...probs, 0) || 1]).range([0, 100]);
  const T = trace.input_token_ids.length - 1;
  const lastLayer = lens.length - 1;

  return (
    <div className="p-3">
      <p className="mb-2 text-[11px] leading-relaxed text-slate-500">
        Each row pushes that block's <span className="value">resid_after_mlp</span> through the
        model's own final LayerNorm and output head, then reads the top-5 tokens at position{' '}
        <span className="value">{T}</span> — showing the top 2 below. The{' '}
        <span className="value">→{lastLayer}</span> row is the model's actual prediction, not a
        projection of it.
      </p>

      {/* Container queries, not viewport breakpoints: this panel also renders in the
          360px grid column, where `sm:`/`md:` would still be "on" and push the
          percent and token-id columns outside the card. */}
      <ol className="@container space-y-0.5">
        {lens.map((entry, i) => (
          <LensRow
            key={entry.layer}
            entry={entry}
            previous={i > 0 ? lens[i - 1] : null}
            isLast={i === lens.length - 1}
            isActive={active === entry.layer}
            barScale={x}
            onSelect={() => selectBlock(entry.layer)}
          />
        ))}
      </ol>
    </div>
  );
}

function LensRow({
  entry,
  previous,
  isLast,
  isActive,
  barScale,
  onSelect,
}: {
  entry: LogitLensEntry;
  previous: LogitLensEntry | null;
  isLast: boolean;
  isActive: boolean;
  barScale: (v: number) => number;
  onSelect: () => void;
}) {
  const top = entry.top_pieces[0] ?? '';
  const prevTop = previous?.top_pieces[0] ?? null;
  // A "swing" is a change of top-1 between adjacent depths. Marking those rows is
  // what turns 12 rows of numbers into a readable progression.
  const swung = previous !== null && prevTop !== top;
  const runnerUp = entry.top_pieces[1] ?? '';

  return (
    <motion.li
      layout="position"
      transition={{ duration: 0.25 }}
      className={[
        'group flex items-center gap-2 rounded px-1.5 py-1',
        isActive ? 'bg-accent-500/10' : 'hover:bg-ink-800',
      ].join(' ')}
    >
      <button
        type="button"
        onClick={onSelect}
        className="flex w-full min-w-0 items-center gap-2 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60"
        title={`open block ${entry.layer + 1}`}
      >
        <span className="value w-7 shrink-0 text-slate-500">
          {isLast ? '→' : 'L'}
          {entry.layer}
        </span>

        <span className="w-3 shrink-0 text-center text-[10px] text-accent-500/70">
          {swung ? '↻' : ''}
        </span>

        <span
          className={[
            'w-24 shrink truncate text-xs @md:w-28',
            isLast ? 'text-accent-400' : 'text-slate-200',
          ].join(' ')}
        >
          {JSON.stringify(top)}
        </span>

        <span className="hidden w-20 shrink-0 truncate text-[11px] text-slate-500 @2xl:block">
          {runnerUp ? JSON.stringify(runnerUp) : ''}
        </span>

        {/* The bar is the only elastic column; everything else is fixed or
            hidden, so the row never exceeds the panel width. */}
        <span className="relative h-3 min-w-0 flex-1 overflow-hidden rounded bg-ink-800">
          <span
            className={[
              'absolute inset-y-0 left-0 rounded',
              isLast ? 'bg-accent-500/70' : 'bg-ink-500',
            ].join(' ')}
            style={{ width: `${barScale(entry.top_probs[0] ?? 0)}%` }}
            aria-hidden
          />
        </span>

        <span className="value w-12 shrink-0 text-right text-[11px] text-slate-300">
          {formatPercent(entry.top_probs[0] ?? 0, 1)}
        </span>

        <span className="value hidden w-14 shrink-0 text-right text-[10px] text-ink-500 @4xl:block">
          #{entry.top_tokens[0]}
        </span>
      </button>
    </motion.li>
  );
}

/** Standalone variant for the compare panel: read-only, no store interaction. */
export function LogitLensStrip({ trace, label }: { trace: Trace | null; label: string }) {
  const lens = trace?.logit_lens ?? [];
  if (lens.length === 0) {
    return <p className="text-[11px] text-slate-500">{label}: no lens</p>;
  }
  return (
    <div>
      <p className="label mb-1">{label}</p>
      <ol className="space-y-0.5">
        {lens.map((entry) => (
          <li key={entry.layer} className="flex items-baseline gap-2 text-[11px]">
            <span className="value w-8 shrink-0 text-slate-500">L{entry.layer}</span>
            <span className="w-24 shrink-0 truncate text-slate-200">
              {JSON.stringify(entry.top_pieces[0] ?? '')}
            </span>
            <span className="value text-slate-400">
              {formatPercent(entry.top_probs[0] ?? 0, 1)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
