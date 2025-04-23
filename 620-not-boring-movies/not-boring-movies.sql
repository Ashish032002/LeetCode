SELECT * FROM Cinema
WHERE id % 2=1 AND Description != "boring" 
ORDER BY rating DESC;