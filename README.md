RAG Imobiliar – Selecție și Estimare de „Preț Corect

Acest proiect implementează un sistem Retrieval-Augmented Generation (RAG) pentru piața imobiliară, care selectează proprietăți relevante și oferă o estimare transparentă a „Prețului Corect” bazată pe proprietăți comparabile (Comps)

Structura Proiectului

data_preprocessing.py -> modul de Date -> Curățare, normalizare și calcul câmpuri derivate.

build_embeddings.py -> modul de Indexare -> Construiește textul de indexare, generează vectori cu all-MiniLM-L6-v2 și populează ChromaDB.

retrieval.py -> modul de Regasire -> Filtrează metadatele și aplică regăsirea (similitudine + filtre logice)

pricing_model.py -> modul de Preț -> Calculează prețul corect prin medie ponderată și stabilește eticheta de preț.

explanation_module.py -> modul de Explicații -> Utilizează un LLM pentru a genera justificarea umană a deciziei.

app.py -> Frontend -> Interfață Streamlit, integrare a modulelor și vizualizare pe hartă (Folium).

Instalare și Rulare

1. Clonare Repozitoriu
2. Creare Mediu Virtual
3. Pregătirea Datelor și Indexarea (rulare data_preprocessing.py si build_embeddings.py)
4. Pornirea Aplicației Streamlit (streamlit run app.py)
