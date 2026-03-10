import { Link } from 'react-router-dom';
import { Camera, CheckCircle, Database, Upload } from 'lucide-react';
import { diseases } from '../data/diseases';
import type { DiseaseImage } from '../types';
import StatsCard from '../components/StatsCard';
import ImageGallery from '../components/ImageGallery';

const mockImages: DiseaseImage[] = [
  {
    id: 'img-001',
    diseaseId: 'ferrugem-asiatica',
    url: '',
    source: 'plantvillage',
    validated: true,
    createdAt: '2026-02-15T10:00:00Z',
  },
  {
    id: 'img-002',
    diseaseId: 'ferrugem-asiatica',
    url: '',
    source: 'kaggle',
    validated: true,
    createdAt: '2026-02-16T11:00:00Z',
  },
  {
    id: 'img-003',
    diseaseId: 'mancha-olho-de-ra',
    url: '',
    source: 'plantvillage',
    validated: true,
    createdAt: '2026-02-17T09:30:00Z',
  },
  {
    id: 'img-004',
    diseaseId: 'mancha-olho-de-ra',
    url: '',
    source: 'field',
    validated: false,
    createdAt: '2026-02-18T14:00:00Z',
  },
  {
    id: 'img-005',
    diseaseId: 'oidio',
    url: '',
    source: 'kaggle',
    validated: true,
    createdAt: '2026-02-19T08:00:00Z',
  },
  {
    id: 'img-006',
    diseaseId: 'oidio',
    url: '',
    source: 'upload',
    validated: false,
    createdAt: '2026-02-20T16:00:00Z',
  },
  {
    id: 'img-007',
    diseaseId: 'antracnose',
    url: '',
    source: 'plantvillage',
    validated: true,
    createdAt: '2026-02-21T12:00:00Z',
  },
  {
    id: 'img-008',
    diseaseId: 'antracnose',
    url: '',
    source: 'field',
    validated: true,
    createdAt: '2026-02-22T10:30:00Z',
  },
  {
    id: 'img-009',
    diseaseId: 'mosaico',
    url: '',
    source: 'kaggle',
    validated: false,
    createdAt: '2026-02-23T15:00:00Z',
  },
  {
    id: 'img-010',
    diseaseId: 'mancha-alvo',
    url: '',
    source: 'plantvillage',
    validated: true,
    createdAt: '2026-02-24T09:00:00Z',
  },
  {
    id: 'img-011',
    diseaseId: 'mancha-alvo',
    url: '',
    source: 'upload',
    validated: false,
    createdAt: '2026-02-25T11:30:00Z',
  },
  {
    id: 'img-012',
    diseaseId: 'folha-saudavel',
    url: '',
    source: 'field',
    validated: true,
    createdAt: '2026-02-26T13:00:00Z',
  },
];

const validatedCount = mockImages.filter((img) => img.validated).length;
const sourcesCount = new Set(mockImages.map((img) => img.source)).size;

export default function Gallery() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text">Banco de Imagens</h1>
          <p className="mt-1 text-text-secondary">
            Imagens coletadas para treinamento do modelo de classificação.
          </p>
        </div>
        <Link
          to="/upload"
          className="flex items-center gap-2 px-5 py-2.5 bg-primary hover:bg-primary-light text-bg font-medium rounded-xl transition-colors"
        >
          <Upload size={18} />
          Upload
        </Link>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatsCard
          title="Total Imagens"
          value={mockImages.length}
          icon={Camera}
          color="bg-primary/15 text-primary"
        />
        <StatsCard
          title="Validadas"
          value={validatedCount}
          icon={CheckCircle}
          color="bg-success/15 text-success"
        />
        <StatsCard
          title="Fontes"
          value={sourcesCount}
          icon={Database}
          color="bg-accent/15 text-accent"
        />
      </div>

      {/* Gallery */}
      <ImageGallery images={mockImages} diseases={diseases} />
    </div>
  );
}
