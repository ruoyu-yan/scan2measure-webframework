import type { QualityTier } from "../types/pipeline";

interface QualityTierSelectProps {
  value: QualityTier;
  onChange: (tier: QualityTier) => void;
  disabled?: boolean;
}

const TIER_INFO: Record<QualityTier, { label: string; time: string }> = {
  preview: { label: "Preview", time: "~2-3 min" },
  balanced: { label: "Balanced", time: "~5-8 min" },
  high: { label: "High", time: "~15-20 min" },
};

export default function QualityTierSelect({ value, onChange, disabled }: QualityTierSelectProps) {
  return (
    <div className="quality-tier-select">
      <label className="quality-tier-select__label">Mesh Quality</label>
      <select
        className="quality-tier-select__select"
        value={value}
        onChange={(e) => onChange(e.target.value as QualityTier)}
        disabled={disabled}
      >
        {(Object.keys(TIER_INFO) as QualityTier[]).map((tier) => (
          <option key={tier} value={tier}>
            {TIER_INFO[tier].label} ({TIER_INFO[tier].time})
          </option>
        ))}
      </select>
    </div>
  );
}
