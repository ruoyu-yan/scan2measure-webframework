interface EntryCardProps {
  title: string;
  description: string;
  icon: string;
  onClick: () => void;
  disabled?: boolean;
}

export default function EntryCard({ title, description, icon, onClick, disabled }: EntryCardProps) {
  return (
    <button
      className={`entry-card${disabled ? " entry-card--disabled" : ""}`}
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
    >
      <div className="entry-card__icon">{icon}</div>
      <h3 className="entry-card__title">{title}</h3>
      <p className="entry-card__desc">{description}</p>
    </button>
  );
}
