from retriever import index_page, search

index_page("t1", 1, "Hemoglobin 9.8 g/dL LOW. Reference 12-16.", "pages/t1_1.png")
index_page("t1", 2, "Patient signature and billing address.", "pages/t1_2.png")
print(search("Is my hemoglobin low?", "t1", k=1))