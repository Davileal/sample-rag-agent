import { Pinecone } from '@pinecone-database/pinecone';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PineconeStore } from '@langchain/pinecone';
import * as dotenv from 'dotenv';

dotenv.config(); // Loads variables from .env

const PDF_PATH = './survery-manual.pdf'; // Path to your PDF file
const CHUNK_SIZE = 1000; // Text chunk size
const CHUNK_OVERLAP = 200; // Overlap between chunks

async function runIngestion() {
    try {
        // --- 1. Initialize Pinecone Client ---
        if (!process.env.PINECONE_API_KEY || !process.env.PINECONE_INDEX) {
            throw new Error(
                'PINECONE_API_KEY and PINECONE_INDEX must be set in the .env file'
            );
        }
        const pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });
        const pineconeIndexName = process.env.PINECONE_INDEX;

        console.log(`Checking if index "${pineconeIndexName}" exists...`);
        const existingIndexes = (await pinecone.listIndexes())?.indexes ?? [];
        if (!existingIndexes.find(index => index.name === pineconeIndexName)) {
            // Log to indicate that the index does not exist and instruct the user to create it.
            // Programmatic creation requires specifying the dimension, which may vary.
            console.error(`Error: Index "${pineconeIndexName}" not found in Pinecone.`);
            console.error(`Please create the index in the Pinecone console with the correct dimension for the embedding model (e.g., ${process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small'} has 1536 dimensions) and 'cosine' metric.`);
            process.exit(1); // Stop execution
        }

        const pineconeIndex = pinecone.Index(pineconeIndexName);
        console.log(`Connected to index "${pineconeIndexName}".`);

        // --- 2. Load and Split the PDF ---
        console.log(`Loading PDF from "${PDF_PATH}"...`);
        const loader = new PDFLoader(PDF_PATH, {
            splitPages: false, // Treats the PDF as a single document
        });
        const docs = await loader.load();
        if (!docs || docs.length === 0) {
            console.error("No documents were loaded from the PDF. Check the file path and content.");
            return;
        }

        // Gets the content of the first page (or the whole document if splitPages=false)
        const pdfText = docs[0].pageContent;
        console.log(`PDF loaded. Total characters: ${pdfText.length}`);

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: CHUNK_SIZE,
            chunkOverlap: CHUNK_OVERLAP,
        });
        const splitDocs = await textSplitter.createDocuments([pdfText]);
        console.log(`Document split into ${splitDocs.length} chunks.`);
        if (splitDocs.length === 0) {
            console.error("No chunks were generated. Check chunk size/overlap settings and PDF content.");
            return;
        }

        // --- 3. Generate Embeddings and Ingest into Pinecone ---
        console.log('Initializing OpenAI embeddings model...');
        const embeddings = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small', // Uses .env or fallback
        });

        console.log(`Inserting ${splitDocs.length} chunks into index "${pineconeIndexName}"...`);
        // Uses PineconeStore.fromDocuments to create embeddings and perform upsert
        await PineconeStore.fromDocuments(splitDocs, embeddings, {
            pineconeIndex,
            maxConcurrency: 5, // Adjust as needed
            namespace: process.env.PINECONE_INDEX_NAMESPACE, // Optional: to organize data within the index
        });

        console.log('--- Ingestion Completed Successfully! ---');
        console.log(`Your Pinecone index "${pineconeIndexName}" now contains the PDF data.`);

    } catch (error) {
        console.error('Error during ingestion:', error);
        // Adds specific details if it's an API Key error
        if (error.message?.includes('Incorrect API key')) {
            console.error("Check if your OPENAI_API_KEY is correct in the .env file");
        }
        if (error.message?.includes('PineconeClient')) {
            console.error("Check if your PINECONE_API_KEY is correct and if the index exists in the correct environment.");
        }
    }
}

runIngestion();
