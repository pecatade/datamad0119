#Challenge 1
#Step 1
SELECT titles.title_id 'TITLE ID', titleauthor.au_id 'AUTHOR ID', titles.price*sales.qty*titles.royalty/100*titleauthor.royaltyper/100 'ROYALTY'
FROM titleauthor
INNER JOIN sales
ON titleauthor.title_id = sales.title_id
INNER JOIN titles
ON titleauthor.title_id = titles.title_id;

#Step 2
SELECT titles.title_id 'TITLE ID', titleauthor.au_id 'AUTHOR ID', SUM(titles.price*sales.qty*titles.royalty/100*titleauthor.royaltyper/100) 'PROFITS'
FROM titleauthor
INNER JOIN sales
ON titleauthor.title_id = sales.title_id
INNER JOIN titles
ON titleauthor.title_id = titles.title_id 
GROUP BY titles.title_id, titleauthor.au_id;

#Step 3
SELECT titleauthor.au_id 'AUTHOR ID', SUM(titles.price*sales.qty*titles.royalty/100*titleauthor.royaltyper/100) 'PROFITS PER AUTHOR'
FROM titleauthor
LEFT JOIN sales
ON titleauthor.title_id = sales.title_id
LEFT JOIN titles
ON titleauthor.title_id = titles.title_id 
GROUP BY titleauthor.au_id
ORDER BY 'PROFITS PER AUTHOR' DESC
LIMIT 3;

#Challenge 2
#Step 1

#CREATE TEMPORARY TABLE royalty_tables
SELECT titles.title_id 'TITLE ID', titleauthor.au_id 'AUTHOR ID', titles.price*sales.qty*titles.royalty/100*titleauthor.royaltyper/100 'ROYALTY'
FROM titles
INNER JOIN titleauthor
ON titles.title_id = titleauthor.title_id
INNER JOIN sales
ON titleauthor.title_id = sales.title_id;

#Step 2
SELECT `TITLE ID`, `AUTHOR ID`, SUM(`ROYALTY`) 'PROFITS'
FROM royalty_tables
GROUP BY `TITLE ID`, `AUTHOR ID`;

#Step 3
SELECT `AUTHOR ID`, SUM(`Royalty`) 'PROFITS PER AUTHOR'
FROM royalty_tables
GROUP BY `AUTHOR ID`
ORDER BY 'PROFITS PER AUTHOR' DESC
LIMIT 3;

#Challenge 3
CREATE TABLE most_profiting_authors
SELECT `AUTHOR ID`, SUM(`Royalty`) 'PROFITS PER AUTHOR'
FROM royalty_tables
GROUP BY `AUTHOR ID`
ORDER BY 'PROFITS PER AUTHOR' DESC
LIMIT 3;