import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutDashboard,
  Search,
  Image,
  Upload,
  Brain,
} from 'lucide-react';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/pesquisa', label: 'Pesquisa', icon: Search },
  { to: '/galeria', label: 'Galeria', icon: Image },
  { to: '/upload', label: 'Upload', icon: Upload },
  { to: '/treinamento', label: 'Treinamento', icon: Brain },
];

export default function Layout() {
  return (
    <div className="flex min-h-screen bg-bg text-text">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-screen w-64 flex flex-col bg-surface border-r border-border">
        {/* Logo */}
        <div className="px-6 py-6">
          <h1 className="text-xl font-bold text-primary">Soja Research</h1>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 space-y-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-primary/10 text-primary'
                    : 'text-text-secondary hover:bg-surface-light hover:text-text'
                }`
              }
            >
              <Icon size={20} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Version indicator */}
        <div className="px-6 py-4 text-xs text-text-secondary">
          v1.0 — Etapa 1
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-64 flex-1 p-8">
        <Outlet />
      </main>
    </div>
  );
}
