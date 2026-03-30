interface EntryCardProps {
  title: string;
  description: string;
  icon: string;
  onClick: () => void;
}

export default function EntryCard({ title, description, icon, onClick }: EntryCardProps) {
  return (
    <button className="entry-card" onClick={onClick}>
      <div className="entry-card__icon">{icon}</div>
      <h3 className="entry-card__title">{title}</h3>
      <p className="entry-card__desc">{description}</p>
    </button>
  );
}
