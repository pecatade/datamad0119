SELECT authors.au_id, authors.au_lname, authors.au_fname, titles.title, publishers.pub_name
FROM authors
INNER JOIN titleauthor
ON authors.au_id = titleauthor.au_id
INNER JOIN titles
ON titles.title_id = titleauthor.title_id
INNER JOIN publishers
ON titles.pub_id = publishers.pub_id