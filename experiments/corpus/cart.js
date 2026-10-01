// Cart totals with progressive discount tiers.
const TIERS = [
  { minSpend: 500, rate: 0.12 },
  { minSpend: 250, rate: 0.07 },
  { minSpend: 0, rate: 0 },
];

export function discountRate(subtotal) {
  const tier = TIERS.find((t) => subtotal >= t.minSpend);
  return tier ? tier.rate : 0;
}

export function applyVat(subtotal, vatRate = 0.2) {
  return round2(subtotal * (1 + vatRate));
}

export function round2(value) {
  return Math.round(value * 100) / 100;
}

export function cartTotal(lines, vatRate) {
  const subtotal = lines.reduce((sum, line) => sum + line.price * line.qty, 0);
  const discounted = subtotal * (1 - discountRate(subtotal));
  return { subtotal: round2(subtotal), total: applyVat(discounted, vatRate) };
}
