require("dotenv").config({ path: require("path").resolve(__dirname, ".env") });

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
const port = Number.parseInt(process.env.PORT || "3001", 10);
const allowedTypes = new Set(["application/pdf", "image/jpeg", "image/jpg", "image/png"]);
const allowedDocumentTypes = new Set(["prescription", "xray"]);
const allowedLanguages = new Set(["en", "te", "hi", "ta"]);
const clientOrigins = (process.env.CLIENT_ORIGINS ||
  "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5500")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);
const deploymentOrigins = [
  process.env.VERCEL_URL && `https://${process.env.VERCEL_URL}`,
  process.env.VERCEL_BRANCH_URL && `https://${process.env.VERCEL_BRANCH_URL}`,
  process.env.VERCEL_PROJECT_PRODUCTION_URL && `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`,
].filter(Boolean);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024, files: 1 },
  fileFilter: (_request, file, callback) => {
    const extension = file.originalname.toLowerCase().split(".").pop();
    const allowedExtension = ["pdf", "jpg", "jpeg", "png"].includes(extension);
    if (!allowedTypes.has(file.mimetype) && !allowedExtension) {
      return callback(new Error("Only PDF, JPEG, and PNG documents are accepted."));
    }
    callback(null, true);
  },
});

app.use(cors({
  origin(origin, callback) {
    // Allow non-browser clients (for example curl) while restricting browsers.
    if (!origin || clientOrigins.includes(origin) || deploymentOrigins.includes(origin)) return callback(null, true);
    return callback(new Error("This browser origin is not allowed."));
  },
}));
app.use(express.json({ limit: "1mb" }));

app.get("/api/health", (_request, response) => {
  response.json({
    ok: true,
    geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
  });
});

function requireGemini() {
  if (!process.env.GEMINI_API_KEY) {
    const error = new Error("Gemini is not configured. Set GEMINI_API_KEY on the server.");
    error.status = 503;
    throw error;
  }
  return new GoogleGenerativeAI(process.env.GEMINI_API_KEY)
    .getGenerativeModel({ model: process.env.GEMINI_MODEL || "gemini-3.6-flash" });
}

const responseSchema = {
  type: "OBJECT",
  properties: {
    summary: { type: "STRING" },
    findings: { type: "ARRAY", items: { type: "STRING" } },
    recommendations: { type: "ARRAY", items: { type: "STRING" } },
    warnings: { type: "ARRAY", items: { type: "STRING" } },
    confidence: { type: "STRING", enum: ["low", "medium", "high"] },
  },
  required: ["summary", "findings", "recommendations", "warnings", "confidence"],
};

function documentPrompt(documentType, language) {
  const focus = documentType === "prescription"
    ? "Extract medicines, dosage, frequency, duration, and prescriber instructions exactly when legible."
    : "Describe visible radiology findings conservatively; do not diagnose or infer beyond the image.";
  const languageInstruction = language === "en"
    ? "Respond in English."
    : `Respond in the user's selected language: ${language}.`;
  return `You are a cautious medical document assistant. Analyze this ${documentType} for informational purposes only.
${focus}
${languageInstruction}
Never invent unreadable text. Clearly say when content cannot be read. Do not provide a diagnosis, treatment change, or emergency triage. Return ONLY valid JSON matching this schema: ${JSON.stringify(responseSchema)}.
Use concise language, include uncertainty in warnings, and recommend review by a qualified clinician.`;
}

app.post("/api/analyze-document", upload.single("document"), async (request, response, next) => {
  try {
    const { documentType, language = "en" } = request.body;
    if (!allowedDocumentTypes.has(documentType)) {
      return response.status(400).json({
        error: "documentType must be either 'prescription' or 'xray'.",
      });
    }
    if (!allowedLanguages.has(language)) {
      return response.status(400).json({ error: "language must be one of 'en', 'te', 'hi', or 'ta'." });
    }
    if (!request.file) {
      return response.status(400).json({ error: "A document file is required in the 'document' field." });
    }

    const model = requireGemini();
    const result = await model.generateContent({
      contents: [{
        role: "user",
        parts: [
          { text: documentPrompt(documentType, language) },
          {
            inlineData: {
              data: request.file.buffer.toString("base64"),
              mimeType: request.file.mimetype,
            },
          },
        ],
      }],
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema,
      },
    });
    const text = result.response.text().replace(/^```json\s*|\s*```$/g, "").trim();
    let analysis;
    try {
      analysis = JSON.parse(text);
    } catch (_error) {
      return response.status(502).json({ error: "Gemini returned an invalid structured response." });
    }
    return response.json({ documentType, analysis });
  } catch (error) {
    return next(error);
  }
});

app.post("/api/chat", async (request, response, next) => {
  try {
    const { message, language = "en" } = request.body || {};
    if (typeof message !== "string" || !message.trim() || message.length > 4000) {
      return response.status(400).json({ error: "message must be a non-empty string up to 4000 characters." });
    }
    if (!allowedLanguages.has(language)) {
      return response.status(400).json({ error: "language must be one of 'en', 'te', 'hi', or 'ta'." });
    }
    const model = requireGemini();
    const result = await model.generateContent(
      `You are MANA-SEHAT's cautious health-information assistant. Answer clearly and briefly in language ${language}.
Do not diagnose, prescribe, or claim certainty. Encourage a qualified clinician for personal medical decisions.
If the user describes an emergency, advise contacting local emergency services.
User message: ${message.trim()}`,
    );
    return response.json({ reply: result.response.text() });
  } catch (error) {
    return next(error);
  }
});

app.use((error, _request, response, _next) => {
  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    return response.status(413).json({ error: "Document exceeds the 10MB limit." });
  }
  if (error.message && (error.message.includes("Only PDF") || error.message.includes("origin"))) {
    return response.status(400).json({ error: error.message });
  }
  const status = Number.isInteger(error.status) ? error.status : 502;
  console.error(error);
  return response.status(status).json({ error: status === 503 ? error.message : "Gemini request failed." });
});

if (require.main === module) {
  app.listen(port, () => {
    console.log(`MANA-SEHAT API listening on port ${port}`);
  });
}

module.exports = app;
