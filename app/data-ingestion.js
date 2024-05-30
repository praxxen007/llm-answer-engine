"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var pinecone_1 = require("@langchain/pinecone");
var openai_1 = require("langchain/embeddings/openai");
var text_splitter_1 = require("langchain/text_splitter");
// import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
var text_1 = require("langchain/document_loaders/fs/text");
var pinecone_2 = require("@pinecone-database/pinecone");
var pc = new pinecone_2.Pinecone({
    apiKey: '4dcf3361-39e2-40ef-aa47-f0d71babe502'
});
function main() {
    return __awaiter(this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, pc.createIndex({
                        name: 'legal',
                        dimension: 1536, // Replace with your model dimensions
                        metric: 'euclidean', // Replace with your model metric
                        spec: {
                            serverless: {
                                cloud: 'aws',
                                region: 'us-east-1'
                            }
                        }
                    })];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    });
}
// const pdfDirectory = 'path/to/pdf/directory';
var pineconeIndex = pc.Index("legal");
function upsert(pdfDirectory) {
    return __awaiter(this, void 0, void 0, function () {
        var pdfLoader, docs, textSplitter, texts, embeddings, pineconeStore;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    pdfLoader = new text_1.TextLoader(pdfDirectory);
                    return [4 /*yield*/, pdfLoader.load()];
                case 1:
                    docs = _a.sent();
                    textSplitter = new text_splitter_1.RecursiveCharacterTextSplitter({ chunkSize: 2000, chunkOverlap: 200 });
                    return [4 /*yield*/, textSplitter.splitDocuments(docs)];
                case 2:
                    texts = _a.sent();
                    embeddings = new openai_1.OpenAIEmbeddings();
                    return [4 /*yield*/, pinecone_1.PineconeStore.fromDocuments(texts, embeddings, {
                            pineconeIndex: pineconeIndex,
                            textKey: 'text',
                            namespace: 'docs',
                            maxConcurrency: 5
                        })];
                case 3:
                    pineconeStore = _a.sent();
                    return [2 /*return*/];
            }
        });
    });
}
main();
upsert("data.txt");
