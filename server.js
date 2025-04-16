import express from 'express';
import * as dotenv from 'dotenv';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { PineconeStore } from '@langchain/pinecone';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000; // Port for the server to run

app.use(express.json()); // Middleware to parse JSON in request bodies

let retrievalChain; // Variable to store the initialized RAG chain

// --- Async Initialization Function ---
async function initializeRagChain() {
    try {
        console.log('Initializing RAG components...');

        // --- 1. Validate Environment Variables ---
        if (!process.env.PINECONE_API_KEY || !process.env.PINECONE_INDEX || !process.env.OPENAI_API_KEY) {
            throw new Error('Environment variables PINECONE_API_KEY, PINECONE_INDEX, and OPENAI_API_KEY are required.');
        }

        // --- 2. Initialize Pinecone Client ---
        const pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });
        const pineconeIndexName = process.env.PINECONE_INDEX;
        const pineconeIndex = pinecone.Index(pineconeIndexName);
        console.log(`Connected to Pinecone index: "${pineconeIndexName}"`);

        // --- 3. Initialize OpenAI Embeddings Model ---
        const embeddings = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small',
        });
        console.log(`OpenAI Embeddings model initialized: "${embeddings.modelName}"`);

        // --- 4. Initialize Vector Store (Pinecone) ---
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex,
            namespace: 'survery-docs-namespace', // Use the same namespace used during ingestion (if any)
        });
        console.log('Pinecone Vector Store initialized.');

        // --- 5. Initialize LLM (OpenAI Chat Model) ---
        const llm = new ChatOpenAI({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: process.env.OPENAI_CHAT_MODEL || 'gpt-3.5-turbo',
            temperature: 0.3, // Adjust to control creativity (0 = more deterministic)
        });
        console.log(`OpenAI Chat model initialized: "${llm.modelName}"`);

        // --- 6. Create the Retriever ---
        // k: number of relevant documents to retrieve from Pinecone
        const retriever = vectorStore.asRetriever({ k: 4 });
        console.log(`Retriever configured to fetch ${retriever.k} documents.`);

        // --- 7. Define the Prompt Template ---
        // This is the core of RAG: instructs the LLM to use retrieved context
        const promptTemplate = `
            You are a virtual assistant specialized in the "Survery" system.
            Your task is to answer user questions about how to use the Survery system.
            Use ONLY the information provided in the CONTEXT below. Do not make up information.
            If the answer is not in the context, politely say that you do not have that specific information in the provided document.
            Respond clearly and concisely in Brazilian Portuguese.

            CONTEXT:
            {context}

            USER QUESTION:
            {input}

            ASSISTANT'S ANSWER:
        `;
        const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);
        console.log('Prompt template defined.');

        // --- 8. Create the Combine Documents Chain ---
        // This chain inserts the retrieved documents into the prompt
        const combineDocsChain = await createStuffDocumentsChain({
            llm: llm,
            prompt: prompt,
        });
        console.log('Combine documents chain created.');

        // --- 9. Create the Retrieval Chain ---
        // This main chain orchestrates: Question -> Retriever -> CombineDocsChain -> LLM Response
        retrievalChain = await createRetrievalChain({
            retriever: retriever,
            combineDocsChain: combineDocsChain,
        });
        console.log('--- RAG Retrieval Chain successfully initialized! ---');

    } catch (error) {
        console.error('--- FATAL ERROR DURING INITIALIZATION ---');
        console.error(error);
        process.exit(1); // Exit the process if initialization fails
    }
}

// --- Chat API Route ---
app.post('/chat', async (req, res) => {
    const { question } = req.body;

    if (!question) {
        return res.status(400).json({ error: 'The "question" property is required in the request body.' });
    }

    if (!retrievalChain) {
        console.error("Error: RAG chain was not initialized. The server may not have started correctly.");
        return res.status(500).json({ error: 'Internal server error: Assistant not initialized.' });
    }

    console.log(`\nReceived question: "${question}"`);

    try {
        console.time('RAG Response Time'); // Measure response time

        // --- Invoke the RAG Chain ---
        const result = await retrievalChain.invoke({
            input: question,
        });

        console.timeEnd('RAG Response Time');

        console.log("Retrieved context:", result.context?.map(doc => doc.pageContent.substring(0, 100) + '...')); // Log summarized context
        console.log("Generated response:", result.answer);

        // --- Send the response to the client ---
        res.json({ answer: result.answer });

    } catch (error) {
        console.error('Error while processing the question:', error);
        res.status(500).json({ error: 'An error occurred while trying to answer your question.' });
    }
});

// --- Health Check Route (Optional) ---
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'OK', timestamp: new Date().toISOString() });
});

// --- Start the Server ---
// First, initialize the RAG chain, then start the Express server
initializeRagChain().then(() => {
    app.listen(port, () => {
        console.log(`\n🚀 Server running on port ${port}`);
        console.log(`Chat endpoint available at POST http://localhost:${port}/chat`);
    });
}).catch(error => {
    // Error already logged in initializeRagChain, just making sure the server doesn't start
    console.error("Failed to start server due to RAG chain initialization error.");
});