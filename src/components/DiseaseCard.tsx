import { Camera, FileText } from 'lucide-react';
import type { Disease } from '../types';

interface DiseaseCardProps {
  disease: Disease;
  onClick?: (disease: Disease) => void;
}

const severityConfig = {
  alta: { label: 'Alta', className: 'bg-error/15 text-error' },
  média: { label: 'Média', className: 'bg-accent/15 text-accent' },
  baixa: { label: 'Baixa', className: 'bg-success/15 text-success' },
};

export default function DiseaseCard({ disease, onClick }: DiseaseCardProps) {
  const severity = severityConfig[disease.severity];

  return (
    <button
      type="button"
      onClick={() => onClick?.(disease)}
      className="w-full text-left bg-surface rounded-xl p-5 hover:bg-surface-light transition-colors cursor-pointer"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{disease.icon}</span>
          <div>
            <h3 className="font-semibold text-text">{disease.name}</h3>
            <p className="text-sm italic text-text-secondary">
              {disease.scientificName}
            </p>
          </div>
        </div>
        <span
          className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${severity.className}`}
        >
          {severity.label}
        </span>
      </div>

      <div className="flex items-center gap-4 mt-4 pt-3 border-t border-border text-sm text-text-secondary">
        <span className="flex items-center gap-1.5">
          <Camera size={14} />
          {disease.imageCount}
        </span>
        <span className="flex items-center gap-1.5">
          <FileText size={14} />
          {disease.articleCount}
        </span>
      </div>
    </button>
  );
}
