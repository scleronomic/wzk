import numpy as np
import bibtexparser
from bibtexparser import middlewares, model

file = "/Users/jote/Documents/BibDesk/phd.bib"

with open(file=file) as f:
    bibtex_string = f.read()

# Lets parse some bibtex string.
bib_database = bibtexparser.parse_string(bibtex_string,
    # Middleware layers to transform parsed entries.
    # Here, we split multiple authors from each other and then extract first name, last name, ... for each
    append_middleware=[middlewares.SeparateCoAuthors(), middlewares.SplitNameParts()],
)


authors = []
for entry in bib_database.blocks:
    if not isinstance(entry, model.Entry):
        continue
    authors_i = entry.fields_dict["author"].value
    for authors_ij in authors_i:
        authors.append(authors_ij.merge_first_name_first)

# print(authors)

authors = sorted(authors)
authors = np.unique(authors)
for a in authors:
    print(a)
