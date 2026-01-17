import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Zap,
  Clock,
  DollarSign,
  Activity,
  ChevronDown,
  Play,
  RefreshCw,
  GitBranch,
  Layers,
  Target,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { routeOrder, benchmarkAlgorithms, fetchQuote, runTCA } from './services/api';

const ALGORITHMS = [
  { value: 'nearest_neighbor', label: 'Nearest Neighbor', complexity: 'O(n²)' },
  { value: 'two_opt', label: '2-Opt Local Search', complexity: 'O(n²·k)' },
  { value: 'simulated_annealing', label: 'Simulated Annealing', complexity: 'O(iter·n)' },
  { value: 'genetic', label: 'Genetic Algorithm', complexity: 'O(gen·pop·n)' },
  { value: 'held_karp', label: 'Held-Karp (Exact)', complexity: 'O(n²·2ⁿ)' },
  { value: 'brute_force', label: 'Brute Force (Exact)', complexity: 'O(n!)' },
];

const EXECUTION_STRATEGIES = [
  { value: 'instant', label: 'Instant', description: 'Execute immediately' },
  { value: 'vwap', label: 'VWAP', description: 'Volume Weighted Avg Price' },
  { value: 'twap', label: 'TWAP', description: 'Time Weighted Avg Price' },
  { value: 'is', label: 'Impl. Shortfall', description: 'Minimize slippage' },
  { value: 'aggressive', label: 'Aggressive', description: 'Front-loaded execution' },
  { value: 'passive', label: 'Passive', description: 'Back-loaded execution' },
];

const VENUE_COLORS = [
  '#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7',
  '#79c0ff', '#7ee787', '#e3b341', '#ff7b72', '#d2a8ff',
];

function formatNumber(num, decimals = 2) {
  if (num === undefined || num === null) return '-';
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function formatBps(bps) {
  if (bps === undefined || bps === null) return '-';
  return `${formatNumber(bps, 2)} bps`;
}

function formatUsd(usd) {
  if (usd === undefined || usd === null) return '-';
  return `$${formatNumber(usd, 2)}`;
}

function StatusBadge({ status }) {
  const styles = {
    idle: 'bg-terminal-border text-terminal-muted',
    loading: 'bg-terminal-yellow/20 text-terminal-yellow',
    success: 'bg-terminal-green/20 text-terminal-green',
    error: 'bg-terminal-red/20 text-terminal-red',
  };

  const labels = {
    idle: 'Ready',
    loading: 'Processing',
    success: 'Complete',
    error: 'Error',
  };

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-mono ${styles[status]}`}>
      {labels[status]}
    </span>
  );
}

function Card({ children, className = '' }) {
  return (
    <div className={`bg-terminal-surface border border-terminal-border rounded-lg ${className}`}>
      {children}
    </div>
  );
}

function Select({ value, onChange, options, className = '' }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`appearance-none bg-terminal-bg border border-terminal-border rounded px-3 py-2 pr-8 text-terminal-text font-mono text-sm focus:outline-none focus:border-terminal-accent ${className}`}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-terminal-muted pointer-events-none" />
    </div>
  );
}

function Input({ value, onChange, type = 'text', placeholder, className = '' }) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={`bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-terminal-text font-mono text-sm focus:outline-none focus:border-terminal-accent ${className}`}
    />
  );
}

function Button({ children, onClick, variant = 'primary', disabled = false, className = '' }) {
  const variants = {
    primary: 'bg-terminal-accent hover:bg-terminal-accent/80 text-white',
    secondary: 'bg-terminal-border hover:bg-terminal-border/80 text-terminal-text',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2 rounded font-medium text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

function MetricCard({ icon: Icon, label, value, subValue, trend }) {
  return (
    <Card className="p-4">
      <div className="flex items-start justify-between">
        <div className="p-2 bg-terminal-accent/10 rounded">
          <Icon className="w-4 h-4 text-terminal-accent" />
        </div>
        {trend !== undefined && (
          <span className={`text-xs font-mono ${trend >= 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
            {trend >= 0 ? '+' : ''}{formatNumber(trend, 1)}%
          </span>
        )}
      </div>
      <div className="mt-3">
        <p className="text-terminal-muted text-xs uppercase tracking-wider">{label}</p>
        <p className="text-xl font-mono font-semibold text-terminal-text mt-1">{value}</p>
        {subValue && <p className="text-terminal-muted text-xs font-mono mt-1">{subValue}</p>}
      </div>
    </Card>
  );
}

function VenueAllocationChart({ routing }) {
  if (!routing || routing.length === 0) return null;

  const data = routing.map((r, i) => ({
    name: r.venue_id,
    value: r.allocation * 100,
    quantity: r.quantity,
    cost: r.estimated_total_cost_usd,
    fill: VENUE_COLORS[i % VENUE_COLORS.length],
  }));

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={80}
            innerRadius={50}
            paddingAngle={2}
          >
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: '#0d1117',
              border: '1px solid #1c2128',
              borderRadius: '8px',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '12px',
              color: '#c9d1d9',
            }}
            itemStyle={{ color: '#c9d1d9' }}
            labelStyle={{ color: '#c9d1d9' }}
            formatter={(value, name, props) => [
              `${formatNumber(value, 1)}% (${props.payload.quantity.toLocaleString()} shares)`,
              props.payload.name,
            ]}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

function CostBreakdownChart({ cost }) {
  if (!cost) return null;

  const data = [
    { name: 'Fees', value: Math.abs(cost.fees_bps), fill: '#58a6ff' },
    { name: 'Spread', value: cost.spread_cost_bps, fill: '#3fb950' },
    { name: 'Impact', value: cost.impact_cost_bps, fill: '#d29922' },
    { name: 'Latency', value: cost.latency_cost_bps, fill: '#f85149' },
  ].filter(d => d.value > 0);

  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#1c2128" />
          <XAxis type="number" stroke="#6e7681" fontSize={10} tickFormatter={(v) => `${v.toFixed(1)}`} />
          <YAxis type="category" dataKey="name" stroke="#6e7681" fontSize={10} width={60} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0d1117',
              border: '1px solid #1c2128',
              borderRadius: '8px',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '12px',
              color: '#c9d1d9',
            }}
            itemStyle={{ color: '#c9d1d9' }}
            labelStyle={{ color: '#c9d1d9' }}
            formatter={(value) => [`${formatNumber(value, 2)} bps`, 'Cost']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function RoutingTable({ routing, allocationMethod }) {
  if (!routing || routing.length === 0) return null;

  return (
    <div className="overflow-x-auto">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-terminal-muted">Allocation method:</span>
        <span className={`text-xs font-mono px-2 py-0.5 rounded ${allocationMethod === 'smart' ? 'bg-terminal-green/20 text-terminal-green' : 'bg-terminal-border text-terminal-muted'}`}>
          {allocationMethod === 'smart' ? 'Smart (optimized)' : 'Equal'}
        </span>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-terminal-border">
            <th className="text-left py-2 px-3 text-terminal-muted font-medium text-xs uppercase tracking-wider">Seq</th>
            <th className="text-left py-2 px-3 text-terminal-muted font-medium text-xs uppercase tracking-wider">Venue</th>
            <th className="text-right py-2 px-3 text-terminal-muted font-medium text-xs uppercase tracking-wider">Allocation</th>
            <th className="text-right py-2 px-3 text-terminal-muted font-medium text-xs uppercase tracking-wider">Quantity</th>
            <th className="text-right py-2 px-3 text-terminal-muted font-medium text-xs uppercase tracking-wider">Est. Cost</th>
          </tr>
        </thead>
        <tbody>
          {routing.map((r, i) => (
            <tr key={i} className="border-b border-terminal-border/50 hover:bg-terminal-border/20 group">
              <td className="py-2 px-3 font-mono text-terminal-muted">{r.execution_sequence}</td>
              <td className="py-2 px-3">
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: VENUE_COLORS[i % VENUE_COLORS.length] }}
                  />
                  <div>
                    <span className="font-mono">{r.venue_id}</span>
                    {r.allocation_reasoning && (
                      <p className="text-xs text-terminal-muted hidden group-hover:block">{r.allocation_reasoning}</p>
                    )}
                  </div>
                </div>
              </td>
              <td className="py-2 px-3 text-right font-mono">{formatNumber(r.allocation * 100, 1)}%</td>
              <td className="py-2 px-3 text-right font-mono">{r.quantity.toLocaleString()}</td>
              <td className="py-2 px-3 text-right font-mono text-terminal-yellow">{formatUsd(r.estimated_total_cost_usd)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AlgorithmMetrics({ metrics }) {
  if (!metrics) return null;

  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="bg-terminal-bg rounded p-3">
        <p className="text-terminal-muted text-xs uppercase">Execution Time</p>
        <p className="font-mono text-lg">{formatNumber(metrics.execution_time_ms, 2)} ms</p>
      </div>
      <div className="bg-terminal-bg rounded p-3">
        <p className="text-terminal-muted text-xs uppercase">Iterations</p>
        <p className="font-mono text-lg">{metrics.iterations?.toLocaleString() || '-'}</p>
      </div>
      {metrics.nodes_explored && (
        <div className="bg-terminal-bg rounded p-3">
          <p className="text-terminal-muted text-xs uppercase">Nodes Explored</p>
          <p className="font-mono text-lg">{metrics.nodes_explored.toLocaleString()}</p>
        </div>
      )}
      {metrics.generations && (
        <div className="bg-terminal-bg rounded p-3">
          <p className="text-terminal-muted text-xs uppercase">Generations</p>
          <p className="font-mono text-lg">{metrics.generations}</p>
        </div>
      )}
    </div>
  );
}

function ExecutionScheduleChart({ schedule }) {
  if (!schedule || !schedule.slices || schedule.slices.length === 0) return null;

  const data = schedule.slices.map((slice, i) => ({
    name: `T${i + 1}`,
    quantity: slice.target_quantity,
    percentage: slice.target_percentage * 100,
    cumulative: slice.cumulative_percentage * 100,
    participation: slice.volume_participation * 100,
  }));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-terminal-bg rounded p-3">
          <p className="text-terminal-muted text-xs uppercase">Strategy</p>
          <p className="font-mono text-lg uppercase">{schedule.strategy}</p>
        </div>
        <div className="bg-terminal-bg rounded p-3">
          <p className="text-terminal-muted text-xs uppercase">Duration</p>
          <p className="font-mono text-lg">{schedule.duration_minutes} min</p>
        </div>
        <div className="bg-terminal-bg rounded p-3">
          <p className="text-terminal-muted text-xs uppercase">Risk Score</p>
          <p className={`font-mono text-lg ${schedule.risk_score > 0.7 ? 'text-terminal-red' : schedule.risk_score > 0.4 ? 'text-terminal-yellow' : 'text-terminal-green'}`}>
            {formatNumber(schedule.risk_score * 100, 0)}%
          </p>
        </div>
      </div>
      
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1c2128" />
            <XAxis dataKey="name" stroke="#6e7681" fontSize={10} />
            <YAxis stroke="#6e7681" fontSize={10} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d1117',
                border: '1px solid #1c2128',
                borderRadius: '8px',
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '12px',
                color: '#c9d1d9',
              }}
              itemStyle={{ color: '#c9d1d9' }}
              labelStyle={{ color: '#c9d1d9' }}
              formatter={(value, name) => {
                if (name === 'quantity') return [value.toLocaleString() + ' shares', 'Quantity'];
                if (name === 'percentage') return [formatNumber(value, 1) + '%', 'Allocation'];
                return [formatNumber(value, 1) + '%', name];
              }}
            />
            <Bar dataKey="quantity" fill="#58a6ff" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="text-xs text-terminal-muted">
        Expected market participation: {formatNumber(schedule.expected_participation_rate * 100, 2)}% of volume
      </div>
    </div>
  );
}

function TCAResults({ tca }) {
  if (!tca) return null;

  const slippageColor = (val) => {
    if (val < 0) return 'text-terminal-green';
    if (val > 2) return 'text-terminal-red';
    return 'text-terminal-yellow';
  };

  const costAttributionData = [
    { name: 'Spread', value: tca.cost_attribution.spread_cost_bps, fill: '#58a6ff' },
    { name: 'Impact', value: tca.cost_attribution.impact_cost_bps, fill: '#3fb950' },
    { name: 'Timing', value: tca.cost_attribution.timing_cost_bps, fill: '#d29922' },
    { name: 'Fees', value: tca.cost_attribution.fee_cost_bps, fill: '#f85149' },
    { name: 'Opportunity', value: tca.cost_attribution.opportunity_cost_bps, fill: '#a371f7' },
  ].filter(d => d.value > 0);

  const fillsOverTime = tca.fills.map((f, i) => ({
    name: `F${i + 1}`,
    price: f.price,
    quantity: f.quantity,
    venue: f.venue_id,
  }));

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Performance Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          icon={TrendingUp}
          label="Impl. Shortfall"
          value={formatBps(tca.implementation_shortfall_bps)}
          subValue={tca.implementation_shortfall_bps < 0 ? 'Outperformed' : 'Underperformed'}
        />
        <MetricCard
          icon={BarChart3}
          label="vs VWAP"
          value={formatBps(tca.vwap_slippage_bps)}
          subValue={tca.vwap_slippage_bps < 0 ? 'Beat VWAP' : 'Missed VWAP'}
        />
        <MetricCard
          icon={Activity}
          label="Fill Rate"
          value={`${formatNumber(tca.fill_rate * 100, 1)}%`}
          subValue={`${tca.filled_quantity.toLocaleString()} filled`}
        />
        <MetricCard
          icon={Zap}
          label="Total Cost"
          value={formatBps(tca.cost_attribution.total_bps)}
          subValue={formatUsd(tca.cost_attribution.total_bps * tca.filled_quantity * tca.average_fill_price / 10000)}
        />
      </div>

      {/* Price Summary */}
      <Card className="p-4">
        <h3 className="text-sm font-medium text-terminal-text mb-4">Price Analysis</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-terminal-bg rounded p-3">
            <p className="text-terminal-muted text-xs uppercase">Arrival Price</p>
            <p className="font-mono text-lg">{formatUsd(tca.arrival_price)}</p>
          </div>
          <div className="bg-terminal-bg rounded p-3">
            <p className="text-terminal-muted text-xs uppercase">Avg Fill Price</p>
            <p className={`font-mono text-lg ${slippageColor(tca.arrival_slippage_bps)}`}>
              {formatUsd(tca.average_fill_price)}
            </p>
          </div>
          <div className="bg-terminal-bg rounded p-3">
            <p className="text-terminal-muted text-xs uppercase">VWAP Benchmark</p>
            <p className="font-mono text-lg">{formatUsd(tca.vwap_benchmark)}</p>
          </div>
          <div className="bg-terminal-bg rounded p-3">
            <p className="text-terminal-muted text-xs uppercase">Final Price</p>
            <p className="font-mono text-lg">{formatUsd(tca.final_price)}</p>
          </div>
        </div>
      </Card>

      {/* Slippage Comparison */}
      <Card className="p-4">
        <h3 className="text-sm font-medium text-terminal-text mb-4">Benchmark Comparison</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-terminal-muted">vs Arrival Price</span>
            <span className={`font-mono ${slippageColor(tca.arrival_slippage_bps)}`}>
              {tca.arrival_slippage_bps >= 0 ? '+' : ''}{formatNumber(tca.arrival_slippage_bps, 2)} bps
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-terminal-muted">vs VWAP</span>
            <span className={`font-mono ${slippageColor(tca.vwap_slippage_bps)}`}>
              {tca.vwap_slippage_bps >= 0 ? '+' : ''}{formatNumber(tca.vwap_slippage_bps, 2)} bps
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-terminal-muted">vs TWAP</span>
            <span className={`font-mono ${slippageColor(tca.twap_slippage_bps)}`}>
              {tca.twap_slippage_bps >= 0 ? '+' : ''}{formatNumber(tca.twap_slippage_bps, 2)} bps
            </span>
          </div>
          <div className="flex items-center justify-between border-t border-terminal-border pt-3 mt-3">
            <span className="text-sm font-medium">Price Improvement</span>
            <span className={`font-mono ${tca.price_improvement_bps > 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
              {tca.price_improvement_bps >= 0 ? '+' : ''}{formatNumber(tca.price_improvement_bps, 2)} bps
            </span>
          </div>
        </div>
      </Card>

      {/* Cost Attribution */}
      <Card className="p-4">
        <h3 className="text-sm font-medium text-terminal-text mb-4">Cost Attribution</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={costAttributionData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={70}
                  innerRadius={40}
                >
                  {costAttributionData.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0d1117',
                    border: '1px solid #1c2128',
                    borderRadius: '8px',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '12px',
                    color: '#c9d1d9',
                  }}
                  itemStyle={{ color: '#c9d1d9' }}
                  formatter={(value) => [`${formatNumber(value, 2)} bps`, '']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2">
            {costAttributionData.map((item) => (
              <div key={item.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: item.fill }} />
                  <span className="text-sm text-terminal-muted">{item.name}</span>
                </div>
                <span className="font-mono text-sm">{formatNumber(item.value, 2)} bps</span>
              </div>
            ))}
            <div className="flex items-center justify-between border-t border-terminal-border pt-2 mt-2">
              <span className="text-sm font-medium">Total</span>
              <span className="font-mono font-medium">{formatNumber(tca.cost_attribution.total_bps, 2)} bps</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Execution Timeline */}
      <Card className="p-4">
        <h3 className="text-sm font-medium text-terminal-text mb-4">Execution Timeline</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={fillsOverTime}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1c2128" />
              <XAxis dataKey="name" stroke="#6e7681" fontSize={10} />
              <YAxis stroke="#6e7681" fontSize={10} domain={['dataMin - 0.1', 'dataMax + 0.1']} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0d1117',
                  border: '1px solid #1c2128',
                  borderRadius: '8px',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '12px',
                  color: '#c9d1d9',
                }}
                itemStyle={{ color: '#c9d1d9' }}
                labelStyle={{ color: '#c9d1d9' }}
                formatter={(value, name, props) => {
                  if (name === 'price') return [formatUsd(value), 'Price'];
                  return [value, name];
                }}
              />
              <Bar dataKey="price" fill="#58a6ff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3 text-xs text-terminal-muted">
          {tca.num_fills} fills over {formatNumber(tca.duration_seconds / 60, 0)} minutes
        </div>
      </Card>

      {/* Venue Performance */}
      <Card className="p-4">
        <h3 className="text-sm font-medium text-terminal-text mb-4">Venue Performance</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-terminal-border">
                <th className="text-left py-2 px-3 text-terminal-muted font-medium text-xs uppercase">Venue</th>
                <th className="text-right py-2 px-3 text-terminal-muted font-medium text-xs uppercase">Quantity</th>
                <th className="text-right py-2 px-3 text-terminal-muted font-medium text-xs uppercase">Avg Price</th>
                <th className="text-right py-2 px-3 text-terminal-muted font-medium text-xs uppercase">Slippage</th>
                <th className="text-right py-2 px-3 text-terminal-muted font-medium text-xs uppercase">Fees</th>
              </tr>
            </thead>
            <tbody>
              {tca.venue_performance.map((v, i) => (
                <tr key={i} className="border-b border-terminal-border/50">
                  <td className="py-2 px-3 font-mono">{v.venue_id}</td>
                  <td className="py-2 px-3 text-right font-mono">{v.quantity.toLocaleString()}</td>
                  <td className="py-2 px-3 text-right font-mono">{formatUsd(v.avg_price)}</td>
                  <td className={`py-2 px-3 text-right font-mono ${slippageColor(v.slippage_bps)}`}>
                    {v.slippage_bps >= 0 ? '+' : ''}{formatNumber(v.slippage_bps, 2)} bps
                  </td>
                  <td className="py-2 px-3 text-right font-mono">{formatUsd(v.fees_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Risk Metrics */}
      <Card className="p-4">
        <h3 className="text-sm font-medium text-terminal-text mb-4">Risk Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-terminal-bg rounded p-3">
            <p className="text-terminal-muted text-xs uppercase">Execution Risk</p>
            <p className={`font-mono text-lg ${tca.execution_risk_score > 0.7 ? 'text-terminal-red' : tca.execution_risk_score > 0.4 ? 'text-terminal-yellow' : 'text-terminal-green'}`}>
              {formatNumber(tca.execution_risk_score * 100, 0)}%
            </p>
          </div>
          <div className="bg-terminal-bg rounded p-3">
            <p className="text-terminal-muted text-xs uppercase">Timing Risk</p>
            <p className="font-mono text-lg">{formatNumber(tca.timing_risk_realized_bps, 2)} bps</p>
          </div>
          <div className="bg-terminal-bg rounded p-3">
            <p className="text-terminal-muted text-xs uppercase">Participation Rate</p>
            <p className="font-mono text-lg">{formatNumber(tca.participation_rate * 100, 2)}%</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default function App() {
  const [symbol, setSymbol] = useState('AAPL');
  const [quantity, setQuantity] = useState('10000');
  const [side, setSide] = useState('buy');
  const [urgency, setUrgency] = useState('medium');
  const [algorithm, setAlgorithm] = useState('simulated_annealing');
  const [includeDarkPools, setIncludeDarkPools] = useState(false);
  const [executionStrategy, setExecutionStrategy] = useState('instant');
  const [durationMinutes, setDurationMinutes] = useState('120');
  const [smartAllocation, setSmartAllocation] = useState(true);

  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [quote, setQuote] = useState(null);
  const [tcaResult, setTcaResult] = useState(null);
  const [activeTab, setActiveTab] = useState('route'); // 'route' or 'tca'

  const handleRoute = async () => {
    setStatus('loading');
    setError(null);
    try {
      const response = await routeOrder({
        symbol: symbol.toUpperCase(),
        quantity: parseInt(quantity, 10),
        side,
        urgency,
        algorithm,
        include_dark_pools: includeDarkPools,
        execution_strategy: executionStrategy,
        duration_minutes: parseInt(durationMinutes, 10),
        smart_allocation: smartAllocation,
      });
      setResult(response);
      setQuote(response.market_data);
      setStatus('success');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  const handleRunTCA = async () => {
    setStatus('loading');
    setError(null);
    try {
      const response = await runTCA({
        symbol: symbol.toUpperCase(),
        quantity: parseInt(quantity, 10),
        side,
        urgency,
        algorithm,
        execution_strategy: executionStrategy,
        duration_minutes: parseInt(durationMinutes, 10),
        smart_allocation: smartAllocation,
      });
      setTcaResult(response);
      setStatus('success');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  const handleRefreshQuote = async () => {
    try {
      const q = await fetchQuote(symbol.toUpperCase());
      setQuote(q);
    } catch (err) {
      console.error('Failed to fetch quote:', err);
    }
  };

  useEffect(() => {
    handleRefreshQuote();
  }, [symbol]);

  return (
    <div className="min-h-screen bg-terminal-bg">
      {/* Header */}
      <header className="border-b border-terminal-border bg-terminal-surface">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-terminal-accent/10 rounded">
                <GitBranch className="w-5 h-5 text-terminal-accent" />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-terminal-text">Smart Order Router</h1>
                <p className="text-xs text-terminal-muted font-mono">TSP Optimization Engine</p>
              </div>
            </div>
            <StatusBadge status={status} />
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Order Input Panel */}
          <div className="col-span-12 lg:col-span-4">
            <Card className="p-4">
              <h2 className="text-sm font-medium text-terminal-text mb-4 flex items-center gap-2">
                <Target className="w-4 h-4 text-terminal-accent" />
                Order Parameters
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-terminal-muted uppercase tracking-wider mb-1">Symbol</label>
                  <Input
                    value={symbol}
                    onChange={setSymbol}
                    placeholder="AAPL"
                    className="w-full uppercase"
                  />
                </div>

                <div>
                  <label className="block text-xs text-terminal-muted uppercase tracking-wider mb-1">Quantity</label>
                  <Input
                    type="number"
                    value={quantity}
                    onChange={setQuantity}
                    placeholder="10000"
                    className="w-full"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-terminal-muted uppercase tracking-wider mb-1">Side</label>
                    <Select
                      value={side}
                      onChange={setSide}
                      options={[
                        { value: 'buy', label: 'Buy' },
                        { value: 'sell', label: 'Sell' },
                      ]}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-terminal-muted uppercase tracking-wider mb-1">Urgency</label>
                    <Select
                      value={urgency}
                      onChange={setUrgency}
                      options={[
                        { value: 'low', label: 'Low' },
                        { value: 'medium', label: 'Medium' },
                        { value: 'high', label: 'High' },
                      ]}
                      className="w-full"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-terminal-muted uppercase tracking-wider mb-1">Algorithm</label>
                  <Select
                    value={algorithm}
                    onChange={setAlgorithm}
                    options={ALGORITHMS}
                    className="w-full"
                  />
                  <p className="text-xs text-terminal-muted mt-1 font-mono">
                    {ALGORITHMS.find(a => a.value === algorithm)?.complexity}
                  </p>
                </div>

                <div>
                  <label className="block text-xs text-terminal-muted uppercase tracking-wider mb-1">Execution Strategy</label>
                  <Select
                    value={executionStrategy}
                    onChange={setExecutionStrategy}
                    options={EXECUTION_STRATEGIES}
                    className="w-full"
                  />
                  <p className="text-xs text-terminal-muted mt-1">
                    {EXECUTION_STRATEGIES.find(s => s.value === executionStrategy)?.description}
                  </p>
                </div>

                {executionStrategy !== 'instant' && (
                  <div>
                    <label className="block text-xs text-terminal-muted uppercase tracking-wider mb-1">Duration (minutes)</label>
                    <Input
                      type="number"
                      value={durationMinutes}
                      onChange={setDurationMinutes}
                      placeholder="120"
                      className="w-full"
                    />
                  </div>
                )}

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="smartAllocation"
                    checked={smartAllocation}
                    onChange={(e) => setSmartAllocation(e.target.checked)}
                    className="rounded border-terminal-border bg-terminal-bg"
                  />
                  <label htmlFor="smartAllocation" className="text-sm text-terminal-muted">
                    Smart allocation (vs equal)
                  </label>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="darkPools"
                    checked={includeDarkPools}
                    onChange={(e) => setIncludeDarkPools(e.target.checked)}
                    className="rounded border-terminal-border bg-terminal-bg"
                  />
                  <label htmlFor="darkPools" className="text-sm text-terminal-muted">
                    Include dark pools
                  </label>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  <Button 
                    onClick={() => { setActiveTab('route'); handleRoute(); }} 
                    disabled={status === 'loading'} 
                    className="flex-1 justify-center"
                    variant={activeTab === 'route' ? 'primary' : 'secondary'}
                  >
                    {status === 'loading' && activeTab === 'route' ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Play className="w-4 h-4" />
                    )}
                    Route
                  </Button>
                  <Button 
                    onClick={() => { setActiveTab('tca'); handleRunTCA(); }} 
                    disabled={status === 'loading'}
                    className="flex-1 justify-center"
                    variant={activeTab === 'tca' ? 'primary' : 'secondary'}
                  >
                    {status === 'loading' && activeTab === 'tca' ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <BarChart3 className="w-4 h-4" />
                    )}
                    TCA
                  </Button>
                </div>

                {error && (
                  <div className="p-3 bg-terminal-red/10 border border-terminal-red/20 rounded text-terminal-red text-sm">
                    {error}
                  </div>
                )}
              </div>
            </Card>

            {/* Quote Card */}
            {quote && (
              <Card className="p-4 mt-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-sm font-medium text-terminal-text flex items-center gap-2">
                    <Activity className="w-4 h-4 text-terminal-accent" />
                    Market Data
                  </h2>
                  <button onClick={handleRefreshQuote} className="text-terminal-muted hover:text-terminal-text">
                    <RefreshCw className="w-3 h-3" />
                  </button>
                </div>
                <div className="space-y-2 font-mono text-sm">
                  <div className="flex justify-between">
                    <span className="text-terminal-muted">Bid</span>
                    <span className="text-terminal-green">{formatUsd(quote.bid)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-terminal-muted">Ask</span>
                    <span className="text-terminal-red">{formatUsd(quote.ask)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-terminal-muted">Spread</span>
                    <span>{formatBps(quote.spread_bps)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-terminal-muted">Volume</span>
                    <span>{quote.volume?.toLocaleString()}</span>
                  </div>
                </div>
              </Card>
            )}
          </div>

          {/* Results Panel */}
          <div className="col-span-12 lg:col-span-8">
            {/* Tab Indicator */}
            {(result || tcaResult) && (
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setActiveTab('route')}
                  className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                    activeTab === 'route' 
                      ? 'bg-terminal-accent text-white' 
                      : 'bg-terminal-border text-terminal-muted hover:text-terminal-text'
                  }`}
                >
                  Routing Plan
                </button>
                <button
                  onClick={() => setActiveTab('tca')}
                  className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                    activeTab === 'tca' 
                      ? 'bg-terminal-accent text-white' 
                      : 'bg-terminal-border text-terminal-muted hover:text-terminal-text'
                  }`}
                >
                  TCA Analysis
                </button>
              </div>
            )}

            {activeTab === 'route' && result ? (
              <div className="space-y-6 animate-slide-up">
                {/* Metrics Row */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    icon={DollarSign}
                    label="Total Cost"
                    value={formatBps(result.cost.total_bps)}
                    subValue={formatUsd(result.cost.total_usd)}
                  />
                  <MetricCard
                    icon={TrendingUp}
                    label="vs Baseline"
                    value={result.savings_vs_baseline_bps ? formatBps(result.savings_vs_baseline_bps) : '-'}
                    subValue="Savings"
                    trend={result.savings_vs_baseline_bps ? (result.savings_vs_baseline_bps / result.baseline_cost?.total_bps * 100) : undefined}
                  />
                  <MetricCard
                    icon={Clock}
                    label="Execution"
                    value={`${formatNumber(result.algorithm_metrics.execution_time_ms, 1)} ms`}
                    subValue={`${result.algorithm_metrics.iterations?.toLocaleString() || '-'} iterations`}
                  />
                  <MetricCard
                    icon={Layers}
                    label="Venues"
                    value={result.routing.length}
                    subValue="Active routes"
                  />
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="p-4">
                    <h3 className="text-sm font-medium text-terminal-text mb-4 flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-terminal-accent" />
                      Venue Allocation
                    </h3>
                    <VenueAllocationChart routing={result.routing} />
                  </Card>

                  <Card className="p-4">
                    <h3 className="text-sm font-medium text-terminal-text mb-4 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-terminal-accent" />
                      Cost Breakdown
                    </h3>
                    <CostBreakdownChart cost={result.cost} />
                  </Card>
                </div>

                {/* Routing Table */}
                <Card className="p-4">
                  <h3 className="text-sm font-medium text-terminal-text mb-4">Routing Details</h3>
                  <RoutingTable routing={result.routing} allocationMethod={result.allocation_method} />
                </Card>

                {/* Execution Schedule (if not instant) */}
                {result.execution_schedule && (
                  <Card className="p-4">
                    <h3 className="text-sm font-medium text-terminal-text mb-4 flex items-center gap-2">
                      <Clock className="w-4 h-4 text-terminal-accent" />
                      Execution Schedule
                    </h3>
                    <ExecutionScheduleChart schedule={result.execution_schedule} />
                  </Card>
                )}

                {/* Algorithm Metrics */}
                <Card className="p-4">
                  <h3 className="text-sm font-medium text-terminal-text mb-4">Algorithm Performance</h3>
                  <AlgorithmMetrics metrics={result.algorithm_metrics} />
                </Card>
              </div>
            ) : activeTab === 'tca' && tcaResult ? (
              <TCAResults tca={tcaResult} />
            ) : (
              <Card className="p-12 flex flex-col items-center justify-center text-center">
                <div className="p-4 bg-terminal-border/30 rounded-full mb-4">
                  <GitBranch className="w-8 h-8 text-terminal-muted" />
                </div>
                <h3 className="text-lg font-medium text-terminal-text mb-2">No Results Yet</h3>
                <p className="text-terminal-muted text-sm max-w-md">
                  Configure your order parameters and click "Route" to see the optimal
                  routing strategy, or "TCA" to run a full execution simulation with
                  transaction cost analysis.
                </p>
              </Card>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-terminal-border mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-xs text-terminal-muted">
            <span>Smart Order Router v1.0.0</span>
            <span className="font-mono">TSP Optimization for Trade Execution</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
