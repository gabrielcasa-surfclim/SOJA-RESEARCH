import { Search, Loader2 } from 'lucide-react';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  placeholder?: string;
  loading?: boolean;
}

export default function SearchBar({
  value,
  onChange,
  onSearch,
  placeholder = 'Pesquisar...',
  loading = false,
}: SearchBarProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch();
  };

  return (
    <form onSubmit={handleSubmit} className="flex w-full gap-2">
      <div className="relative flex-1">
        <Search
          size={18}
          className="absolute left-4 top-1/2 -translate-y-1/2 text-text-secondary"
        />
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-11 pr-4 py-3 bg-surface-light border border-border rounded-xl text-text placeholder:text-text-secondary/50 focus:outline-none focus:border-primary transition-colors"
        />
      </div>
      <button
        type="submit"
        disabled={loading}
        className="px-6 py-3 bg-primary hover:bg-primary-light text-bg font-medium rounded-xl transition-colors disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2"
      >
        {loading ? (
          <Loader2 size={18} className="animate-spin" />
        ) : (
          <Search size={18} />
        )}
        Buscar
      </button>
    </form>
  );
}
