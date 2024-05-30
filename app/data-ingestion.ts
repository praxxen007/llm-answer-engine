import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
// import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { TextLoader } from 'langchain/document_loaders/fs/text';

import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone({
  apiKey: '4dcf3361-39e2-40ef-aa47-f0d71babe502'
});


async function main() {

  await pc.createIndex({
    name: 'legal',
    dimension: 1536, // Replace with your model dimensions
    metric: 'euclidean', // Replace with your model metric
    spec: {
      serverless: {
        cloud: 'aws',
        region: 'us-east-1'
      }
    }
  });
}

// const pdfDirectory = 'path/to/pdf/directory';
const pineconeIndex = pc.Index("legal");

async function upsert(pdfDirectory:string){
  const pdfLoader = new TextLoader(pdfDirectory);
  const docs = await pdfLoader.load();

  // Split documents into smaller chunks for better indexing
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 2000, chunkOverlap: 200 });
  const texts = await textSplitter.splitDocuments(docs);

  // Create embeddings for the text chunks
  const embeddings = new OpenAIEmbeddings();


  // Create a Pinecone vector store instance
  const pineconeStore = await PineconeStore.fromDocuments(texts, embeddings, {
    pineconeIndex,
    textKey: 'text',
    namespace: 'docs',
    maxConcurrency: 5
  });

  // Ingest data into Pinecone
  // await PineconeStore.fromDocuments(docs, new OpenAIEmbeddings(), {
  //   pineconeIndex,
  //   maxConcurrency: 5, // Maximum number of batch requests to allow at once. Each batch is 1000 vectors.
  // });
}
main();

upsert("data.txt")