SELECT authors.au_id 'AUTHOR ID', authors.au_lname 'LAST NAME', authors.au_fname 'FIRST NAME', titles.price * sales.qty * titles.royalty / 100 * titleauthor.royaltyper / 100 AS PROFIT
FROM authors
LEFT JOIN titleauthor
ON authors.au_id = titleauthor.au_id
LEFT JOIN sales
ON titleauthor.title_id = sales.title_id
LEFT JOIN titles
ON titles.title_id = titleauthor.title_id
GROUP BY AUTHORS.AU_ID
