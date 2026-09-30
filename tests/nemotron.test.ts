import { describe, it, expect } from "vitest";
import { callAIProvider } from "../server";

describe("Nemotron Provider Payloads", () => {
  it("NVIDIABuild-Autogen-33 model", async () => {
    let lastFetchCall: { url: string; options: any } | null = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string, options: any) => {
      lastFetchCall = { url, options };
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            choices: [
              {
                message: {
                  content: "Nemotron Response",
                  reasoning_content: "Thought process for GPU computing",
                },
              },
            ],
          }),
      };
    }) as any;

    try {
      const resultAutogen = await callAIProvider({
        provider: "nemotron",
        apiKey: "nvapi-test-key",
        model: "NVIDIABuild-Autogen-33",
        systemInstruction: "You are a helpful assistant.",
        prompt: "Test prompt for Autogen model",
      });

      expect(lastFetchCall).toBeTruthy();
      expect(lastFetchCall!.url).toBe("https://integrate.api.nvidia.com/v1/chat/completions");
      const autogenBody = JSON.parse(lastFetchCall!.options.body);
      expect(autogenBody.model).toBe("NVIDIABuild-Autogen-33");
      expect(resultAutogen.model).toBe("NVIDIABuild-Autogen-33");
      expect(resultAutogen.provider).toBe("nemotron");
      expect(resultAutogen.text).toBe("Nemotron Response");
      expect(lastFetchCall!.options.headers["Authorization"]).toBe("Bearer nvapi-test-key");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("Default Nemotron model (nvidia/nemotron-3.5-lightning-30b-a3b)", async () => {
    let lastFetchCall: { url: string; options: any } | null = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string, options: any) => {
      lastFetchCall = { url, options };
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            choices: [
              {
                message: {
                  content: "Nemotron Response",
                  reasoning_content: "Thought process for GPU computing",
                },
              },
            ],
          }),
      };
    }) as any;

    try {
      const res1 = await callAIProvider({
        provider: "nemotron",
        apiKey: "nvapi-test-key",
        systemInstruction: "You are a helpful assistant.",
        prompt: "Write a limerick about GPU computing",
      });

      expect(lastFetchCall).toBeTruthy();
      expect(lastFetchCall!.url).toBe("https://integrate.api.nvidia.com/v1/chat/completions");

      const body1 = JSON.parse(lastFetchCall!.options.body);
      expect(body1.model).toBe("nvidia/nemotron-3.5-lightning-30b-a3b");
      expect(body1.max_tokens).toBe(16384);
      expect(body1.extra_body).toEqual({
        chat_template_kwargs: { enable_thinking: true },
        reasoning_budget: 16384,
      });
      expect(res1.model).toBe("nvidia/nemotron-3.5-lightning-30b-a3b");
      expect(res1.text).toBe("Nemotron Response");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("Legacy Nemotron model (nvidia/llama-3.1-nemotron-70b-instruct)", async () => {
    let lastFetchCall: { url: string; options: any } | null = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string, options: any) => {
      lastFetchCall = { url, options };
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            choices: [
              {
                message: {
                  content: "Nemotron Response",
                  reasoning_content: "Thought process for GPU computing",
                },
              },
            ],
          }),
      };
    }) as any;

    try {
      const res2 = await callAIProvider({
        provider: "nemotron",
        apiKey: "nvapi-test-key",
        model: "nvidia/llama-3.1-nemotron-70b-instruct",
        systemInstruction: "You are a helpful assistant.",
        prompt: "Hello Nemotron 70B",
      });

      expect(lastFetchCall).toBeTruthy();
      const body2 = JSON.parse(lastFetchCall!.options.body);
      expect(body2.model).toBe("nvidia/llama-3.1-nemotron-70b-instruct");
      expect(body2.max_tokens).toBe(4096);
      expect(body2.extra_body).toBeUndefined();
      expect(res2.model).toBe("nvidia/llama-3.1-nemotron-70b-instruct");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
