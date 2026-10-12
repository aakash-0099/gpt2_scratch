/**
 * The sampling distribution, and the token that was drawn from it.
 *
 * The candidates are the *post-temperature, post-top-k* set, with probabilities
 * recomputed after temperature — i.e. exactly the distribution the draw came
 * from, which is why they sum to 1 over the filtered set. The sampled token is
 * highlighted so the randomness is visibly a draw from a visible distribution
 * rather than a number that appeared from nowhere.
 *
 * Note what is *not* here: the pre-filter logits for the ~50,000 tokens that
 * top-k discarded. Showing them would need the full vocabulary row, which the
 * contract deliberately does not ship.
 */

import { scaleLinear } from 'd3-scale';
import { motion } from 'framer-motion';

import { formatCompact, formatPercent, formatTemperature, formatTopK } from '@/lib/format';
import type { SamplingStep, Trace } from '@/types/trace';

export function TopKChart({ trace }: { trace: Trace | null }) {
  const sampling: SamplingStep | null = trace?.sampling ?? null;

  if (!trace) {
    return (
      <div className="flex h-full items-center justify-center px-3 py-8 text-xs text-slate-500">
        no trace yet
      </div>
    );
  }

  if (!sampling) {
    return (
      <div className="px-3 py-4">
        <p className="text-[11px] leading-relaxed text-slate-500">
          This trace came from a single traced forward pass, not from generation, so there is no
          sampling step to show. Its top-5 prediction at the last position is:
        </p>
        <ol className="mt-2 space-y-1">
          {trace.top_pieces.map((piece, i) => (
            <li key={trace.top_tokens[i]} className="flex items-baseline gap-2 text-[11px]">
              <span className="value w-5 shrink-0 text-slate-500">{i + 1}</span>
              <span className="w-28 shrink-0 truncate text-slate-200">{JSON.stringify(piece)}</span>
              <span className="value text-slate-400">{formatPercent(trace.top_probs[i] ?? 0, 2)}</span>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  const candidates = sampling.candidates;
  const probs = candidates.map((c) => c.prob);
  const hi = Math.max(...probs, 0) || 1;
  const x = scaleLinear().domain([0, hi]).range([0, 100]);
  const sum = probs.reduce((a, b) => a + b, 0);

  // Candidates arrive sorted by probability, but sorting explicitly keeps the
  // chart honest if a future backend stops guaranteeing the order.
  const ranked = [...candidates].sort((a, b) => b.prob - a.prob);
  const sampled = ranked.findIndex((c) => c.token_id === sampling.sampled_token_id);

  return (
    <div className="p-3">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[11px]">
        <span className="label">
          step <span className="value text-slate-300">{sampling.step}</span>
        </span>
        <span className="label">
          temperature{' '}
          <span className="value text-slate-300">{formatTemperature(sampling.temperature)}</span>
        </span>
        <span className="label">
          top-k <span className="value text-slate-300">{formatTopK(sampling.top_k)}</span>
        </span>
        <span className="label">
          candidates <span className="value text-slate-300">{candidates.length}</span>
        </span>
        <span className="label">
          Σ prob <span className="value text-slate-300">{formatPercent(sum, 2)}</span>
        </span>
        {sampling.greedy && (
          <span className="text-[10px] text-amber-400/90">
            greedy: argmax, the distribution below is shown but not sampled from
          </span>
        )}
      </div>

      <motion.ol
        key={`${sampling.step}-${candidates.length}`}
        className="space-y-[3px]"
        initial="hidden"
        animate="shown"
        variants={{ shown: { transition: { staggerChildren: 0.008 } } }}
      >
        {ranked.map((c, i) => {
          const isSampled = c.token_id === sampling.sampled_token_id;
          return (
            <motion.li
              key={c.token_id}
              variants={{ hidden: { opacity: 0, x: -6 }, shown: { opacity: 1, x: 0 } }}
              transition={{ duration: 0.2 }}
              className={[
                'relative flex h-5 items-center gap-2 overflow-hidden rounded px-1.5',
                isSampled ? 'ring-1 ring-accent-500' : '',
              ].join(' ')}
            >
              {/* Probability bar as the row background, so the chart reads at a glance. */}
              <span
                className={[
                  'absolute inset-y-0 left-0 rounded',
                  isSampled ? 'bg-accent-500/35' : 'bg-ink-700',
                ].join(' ')}
                style={{ width: `${x(c.prob)}%` }}
                aria-hidden
              />
              <span className="value relative w-5 shrink-0 text-[10px] text-slate-500">
                {i === 0 ? '①' : i + 1}
              </span>
              <span
                className={[
                  'relative w-32 shrink-0 truncate text-xs',
                  isSampled ? 'text-accent-300' : 'text-slate-200',
                ].join(' ')}
              >
                {JSON.stringify(c.piece)}
              </span>
              <span className="value relative w-16 shrink-0 text-[10px] text-slate-500">
                #{c.token_id}
              </span>
              <span className="relative min-w-[2rem] flex-1" />
              <span className="value relative w-14 shrink-0 text-right text-[10px] text-slate-400">
                {formatCompact(c.logit, 1)}
              </span>
              <span
                className={[
                  'value relative w-12 shrink-0 text-right text-[11px]',
                  isSampled ? 'text-accent-400' : 'text-slate-300',
                ].join(' ')}
              >
                {formatPercent(c.prob, 2)}
              </span>
              {isSampled && (
                <span className="relative shrink-0 text-[10px] font-semibold text-accent-400">
                  sampled
                </span>
              )}
            </motion.li>
          );
        })}
      </motion.ol>

      {ranked.length > 14 && (
        <p className="mt-1.5 text-[10px] text-slate-500">
          showing all {ranked.length} filtered candidates; the rest of the vocabulary was removed by
          top-k{formatTopK(sampling.top_k) === 'off' ? '' : `=${sampling.top_k}`} before
          temperature.
        </p>
      )}

      {sampled >= 0 && (
        <motion.p
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="mt-2 text-[11px] text-slate-400"
        >
          drew rank{' '}
          <span className="value text-accent-400">{sampled + 1}</span> of{' '}
          <span className="value">{ranked.length}</span> —{' '}
          <span className="value text-accent-400">{formatPercent(sampling.sampled_prob, 2)}</span>{' '}
          chance, then appended to the context. The next forward pass runs on{' '}
          <span className="value">{sampling.step + 2}</span> tokens, which is why attention and
          the logit lens change shape every step.
        </motion.p>
      )}

      {sampling.eos_hit && (
        <p className="mt-2 text-[11px] text-amber-400/90">
          this step sampled EOS (id <span className="value">{sampling.eos_id ?? '—'}</span>), so
          generation stopped and EOS was not appended to the sequence.
        </p>
      )}
    </div>
  );
}
