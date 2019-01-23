# Challenge 1
#SELECT authors.au_id 'AUTHOR ID', authors.au_lname 'LAST NAME', authors.au_fname 'FIRST NAME', titles.title 'TITLE', publishers.pub_name 'PUBLISHER'
#FROM authors
#INNER JOIN titleauthor
#ON authors.au_id = titleauthor.au_id
#INNER JOIN titles
#ON titles.title_id = titleauthor.title_id
#INNER JOIN publishers
#ON titles.pub_id = publishers.pub_id;

# Challenge 2
#SELECT authors.au_id 'AUTHOR ID', authors.au_lname 'LAST NAME', authors.au_fname 'FIRST NAME', titles.title 'TITLE', publishers.pub_name 'PUBLISHER', 
#COUNT(titles.title) AS 'TITLE COUNT'
#FROM authors
#INNER JOIN titleauthor
#ON authors.au_id = titleauthor.au_id
#INNER JOIN titles
#ON titles.title_id = titleauthor.title_id
#INNER JOIN publishers
#ON titles.pub_id = publishers.pub_id
#GROUP BY authors.au_id

#Challenge 3
#SELECT authors.au_id 'AUTHOR ID', authors.au_lname 'LAST NAME', authors.au_fname 'FIRST NAME', SUM(sales.qty) AS TOTAL
#FROM authors
#LEFT JOIN titleauthor
#ON authors.au_id = titleauthor.au_id
#LEFT JOIN sales
#ON titleauthor.title_id = sales.title_id
#GROUP BY AUTHORS.AU_ID
#ORDER BY TOTAL DESC
#LIMIT 3;

#Challenge 4
#SELECT authors.au_id 'AUTHOR ID', authors.au_lname 'LAST NAME', authors.au_fname 'FIRST NAME', SUM(sales.qty) AS TOTAL
#FROM authors
#LEFT JOIN titleauthor
#ON authors.au_id = titleauthor.au_id
#LEFT JOIN sales
#ON titleauthor.title_id = sales.title_id
#GROUP BY AUTHORS.AU_ID
#ORDER BY TOTAL DESC
