from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from deep_translator import GoogleTranslator
import re, os

# 🧩 Detect output language from query (like “in Hindi” or “in Telugu”)
def detect_output_language(query):
    match = re.search(r"in (\w+)", query, re.IGNORECASE)
    if match:
        lang_word = match.group(1).lower()
        lang_map = {
            "english": "en",
            "hindi": "hi",
            "kannada": "kn",
            "tamil": "ta",
            "telugu": "te",
            "malayalam": "ml",
            "marathi": "mr",
            "bengali": "bn",
            "gujarati": "gu",
            "urdu": "ur"
        }
        return lang_map.get(lang_word, "en")
    return "en"  # default → English


# 🌐 Translate English answer → target language using Deep Translator
def translate_answer(answer, target_lang="en"):
    if target_lang == "en":
        return answer
    try:
        translated = GoogleTranslator(source="en", target=target_lang).translate(answer)
        print(f"🌍 Translated answer → {target_lang}: {translated}")
        return translated
    except Exception as e:
        print("⚠️ Translation error:", e)
        return answer


# 🧩 Step 1: Load both TXT and PDF legal documents (Constitution, Acts, etc.)
def load_legal_docs(folder_path="data"):
    """Load all .txt and .pdf files from the given folder."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    return documents


# 🧠 Step 2: Build the retrieval-based chatbot chain (Multilingual + Context-only)
def build_legal_chain():
    print("🔍 Loading and preparing legal documents...")
    documents = load_legal_docs("data")

    if not documents:
        raise ValueError("⚠️ No legal documents found in the 'data' folder. Please add .txt or .pdf files.")

    # ✅ Better chunking for legal articles
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_documents(documents)

    print("📚 Creating multilingual embeddings...")
    # ✅ Use multilingual embeddings for Hindi, Telugu, etc.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # ✅ Use LLaMA model (smaller version preferred for GTX 1650)
    llm = Ollama(model="deepseek-v3.1:671b-cloud", num_ctx=8192)

    # If you have enough VRAM, you can try: llm = Ollama(model="llama3:instruct", num_ctx=2048)

    # ⚖️ Strict prompt to avoid irrelevant (US) answers
       # ⚖️ Smarter but India-focused legal reasoning prompt
    strict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an advanced constitutional law assistant and legal expert specializing exclusively "
        "in the Constitution of India and Indian laws.\n\n"
        "Instructions:\n"
        "1️⃣ Use the given context primarily to answer factually and truthfully.\n"
        "2️⃣ If the question is indirectly related or not explicitly covered in the context, "
        "apply your deep reasoning and general knowledge of INDIAN LAW to answer accurately.\n"
        "3️⃣ Always ensure that your answer strictly pertains to Indian legal systems, acts, amendments, "
        "articles, and judicial practices.\n"
        "4️⃣ Never discuss or compare with foreign countries or laws unless it helps clarify Indian context.\n"
        "5️⃣ If absolutely no relevant information is available, respond with:\n"
        "'The provided context does not contain this information, but under Indian law, it can be interpreted as follows...' "
        "and then give a reasoned Indian legal explanation if possible.\n"
        "6️⃣ Be precise, lawful, and formal — avoid speculation or personal opinions.\n\n"
        "Important stylistic rule: Do NOT state or imply whether any part of the answer 'comes from' the provided context "
        "or 'comes from' the assistant's internal knowledge. Present conclusions, reasoning and citations seamlessly — "
        "do not include meta-statements about source or provenance of individual sentences.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Now provide a detailed and well-reasoned answer relevant ONLY to Indian law and the Constitution of India."
    )
)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": strict_prompt}
    )

    print("✅ Legal multilingual chatbot is ready!")
    return chain


# 💬 Step 3: Example of usage
if __name__ == "__main__":
    qa_chain = build_legal_chain()

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Detect target language (e.g. "in Telugu", "in Hindi")
        target_lang = detect_output_language(query)
        print(f"🈯 Detected target language: {target_lang}")

        result = qa_chain.invoke({"question": query, "chat_history": []})
        answer = result["answer"]

        # Translate answer to target language
        translated_answer = translate_answer(answer, target_lang)
        print(f"\nBot ({target_lang}): {translated_answer}")
