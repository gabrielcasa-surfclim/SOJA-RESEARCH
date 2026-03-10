import type { LucideIcon } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: string;
  color?: string;
}

export default function StatsCard({
  title,
  value,
  icon: Icon,
  trend,
  color = 'bg-primary/15 text-primary',
}: StatsCardProps) {
  return (
    <div className="bg-surface rounded-xl p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-3xl font-bold text-text">{value}</p>
          <p className="mt-1 text-sm text-text-secondary">{title}</p>
          {trend && (
            <p className="mt-2 text-xs text-primary-light">{trend}</p>
          )}
        </div>
        <div className={`p-3 rounded-full ${color}`}>
          <Icon size={22} />
        </div>
      </div>
    </div>
  );
}
