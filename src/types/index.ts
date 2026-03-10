export interface Disease {
  id: string;
  name: string;
  scientificName: string;
  symptoms: string[];
  severity: 'baixa' | 'média' | 'alta';
  conditions: string;
  management: string[];
  fungicides: string[];
  imageCount: number;
  articleCount: number;
  icon: string;
}

export interface DiseaseImage {
  id: string;
  diseaseId: string;
  url: string;
  thumbnailUrl?: string;
  source: 'plantvillage' | 'kaggle' | 'field' | 'upload';
  metadata?: {
    stage?: string;
    weather?: string;
    location?: string;
    validatedBy?: string;
  };
  validated: boolean;
  createdAt: string;
}

export interface Article {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  doi?: string;
  source: 'pubmed' | 'crossref' | 'semanticscholar';
  url?: string;
  publishedDate?: string;
  methods?: string[];
  accuracy?: number;
  datasetUsed?: string;
  summary?: string;
  citationCount?: number;
  createdAt: string;
}

export interface ArticleDisease {
  articleId: string;
  diseaseId: string;
  relevanceScore: number;
}

export interface ImageFeedback {
  id: string;
  imageId: string;
  originalDiseaseId: string;
  suggestedDiseaseId: string;
  confidence: 'certeza' | 'suspeita' | 'não sei';
  notes?: string;
  gpsLat?: number;
  gpsLng?: number;
  weather?: 'sol' | 'nublado' | 'chuva';
  createdAt: string;
}

export interface TrainingExperiment {
  id: string;
  modelName: string;
  config: Record<string, unknown>;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    confusionMatrix?: number[][];
  };
  status: 'pending' | 'running' | 'completed' | 'failed';
  commitHash?: string;
  valAccuracy?: number;
  startedAt?: string;
  completedAt?: string;
  createdAt: string;
}

export interface SearchResult {
  source: 'pubmed' | 'crossref' | 'semanticscholar';
  articles: Article[];
  total: number;
}
