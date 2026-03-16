import type { ViewId } from "./types";
import { VIEWS } from "./types";

interface ViewSelectorProps {
  active: ViewId;
  onChange: (id: ViewId) => void;
}

export function ViewSelector({ active, onChange }: ViewSelectorProps) {
  return (
    <fieldset className="mb-3 border border-zinc-800 rounded p-2">
      <legend className="text-[0.65rem] font-semibold uppercase tracking-wider text-zinc-400 px-1">
        View
      </legend>
      <div className="flex flex-wrap gap-1">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            onClick={() => onChange(v.id)}
            className={`text-[0.65rem] px-2 py-0.5 rounded-full border transition-colors ${
              active === v.id
                ? "border-emerald-500 bg-emerald-500/15 text-emerald-400"
                : "border-zinc-700 text-zinc-500 hover:border-zinc-500 hover:text-zinc-300"
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}
