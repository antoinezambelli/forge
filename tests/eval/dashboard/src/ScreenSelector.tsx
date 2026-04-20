import type { ScreenId } from "./types";
import { SCREENS } from "./types";

interface ScreenSelectorProps {
  active: ScreenId;
  onChange: (id: ScreenId) => void;
}

/** Top-level screen selector — segmented control.
 *
 * Visually distinct from the view/scenario pills: connected buttons,
 * no gaps, reads as "pick one of three modes" not "filter tags".
 */
export function ScreenSelector({ active, onChange }: ScreenSelectorProps) {
  return (
    <fieldset className="mb-4">
      <legend className="text-[0.65rem] font-semibold uppercase tracking-wider text-zinc-400 px-1 mb-1">
        Screen
      </legend>
      <div className="flex flex-col rounded border border-zinc-700 overflow-hidden">
        {SCREENS.map((s, i) => (
          <button
            key={s.id}
            onClick={() => onChange(s.id)}
            className={`text-xs px-2 py-1.5 text-left transition-colors ${
              i > 0 ? "border-t border-zinc-700" : ""
            } ${
              active === s.id
                ? "bg-emerald-500/20 text-emerald-300 font-medium"
                : "bg-zinc-900/40 text-zinc-400 hover:bg-zinc-900/70 hover:text-zinc-200"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}
