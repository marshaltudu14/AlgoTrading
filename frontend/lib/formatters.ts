
export function formatIndianCurrency(num: number): string {
  if (num === null || num === undefined) {
    return "N/A";
  }

  const absNum = Math.abs(num);

  if (absNum >= 10000000) { // Crores (1 Crore = 10,000,000)
    return (num / 10000000).toFixed(2) + " Cr";
  }
  if (absNum >= 100000) { // Lakhs (1 Lakh = 100,000)
    return (num / 100000).toFixed(2) + " L";
  }
  if (absNum >= 1000) { // Thousands
    return (num / 1000).toFixed(2) + " K";
  }
  return num.toFixed(2); // Default to 2 decimal places for smaller numbers
}
