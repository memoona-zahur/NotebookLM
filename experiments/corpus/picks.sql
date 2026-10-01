-- Warehouse picking performance by zone.
SELECT
    z.zone_name,
    COUNT(p.pick_id) AS picks,
    ROUND(AVG(EXTRACT(EPOCH FROM (p.completed_at - p.started_at))), 1) AS avg_seconds,
    SUM(CASE WHEN p.completed_at IS NULL THEN 1 ELSE 0 END) AS abandoned
FROM picks p
JOIN zones z ON z.zone_id = p.zone_id
WHERE p.started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY z.zone_name
HAVING COUNT(p.pick_id) > 50
ORDER BY avg_seconds DESC;
