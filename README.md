# 📚 PDF Chatbot RAG (Retrieval-Augmented Generation)

A simple Node.js project that implements **RAG (Retrieval-Augmented Generation)** using a PDF document as a knowledge base. The content is embedded with OpenAI, stored in Pinecone, and queried via a chatbot API using LangChain.

> Ask questions about your documents via a smart assistant powered by GPT.

---

## ✨ Features

- 🔍 **Document Ingestion**: Parse a PDF and split it into chunks using `langchain`.
- 🧠 **Embeddings**: Generate vector embeddings with OpenAI models.
- 🧺 **Vector Store**: Store vectors in Pinecone for semantic search.
- 💬 **Chat API**: Query the assistant via a `/chat` endpoint using RAG.
- 🌎 **Language**: Assistant responds in Brazilian Portuguese (you can customize it).

---

## 📁 Project Structure

```
.
├── ingest.js             # PDF ingestion and embedding into Pinecone
├── server.js             # Express API with RAG chain setup
├── .env                  # Environment variables
├── package.json
└── example-doc.pdf       # Example PDF used as source knowledge
```

> ⚠️ Do **not** commit private or sensitive PDFs to your repository.

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Davileal/sample-rag-agent.git
cd sample-rag-agent
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Environment Variables

Create a `.env` file in the root with the following content:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-3.5-turbo
PORT=3000
```

> ⚠️ Make sure your Pinecone index exists and matches the correct embedding dimensions (e.g., 1536 for `text-embedding-3-small`).

---

## 📄 Ingest Your PDF

Replace `example-doc.pdf` with your own document.

Then run:

```bash
npm run ingest
```

This will:
- Load the PDF.
- Split it into chunks.
- Generate embeddings.
- Store them in Pinecone under the namespace `survery-docs-namespace`.

---

## 🚀 Run the Server

Start the API:

```bash
npm run start
```

You should see:

```bash
🚀 Server running on port 3000
Chat endpoint available at POST http://localhost:3000/chat
```

---

## 📡 API Usage

### POST `/chat`

Ask a question based on the PDF content.

**Request Body:**

```json
{
  "question": "What is the purpose of this platform?"
}
```

**Response:**

```json
{
  "answer": "The platform is designed to help users analyze their data using AI-powered tools..."
}
```

---

## 🧪 Health Check

You can test the server status with:

```bash
GET /health
```

Returns:

```json
{
  "status": "OK",
  "timestamp": "2025-04-16T18:00:00Z"
}
```

---

## 📌 TODO

- [ ] Support for multiple PDF sources
- [ ] Support multilingual responses

---

## 📝 License

MIT — Free to use and modify.

---

## 👨‍💻 Author

Made with ❤️ by [Davi Leal]