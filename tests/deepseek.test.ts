import { describe, it, expect, vi } from "vitest";
import { callAIProvider } from "../server";

describe("DeepSeek Provider Payloads", () => {
  it("deepseek-chat model (standard system & user messages)", async () => {
    let lastFetchCall: { url: string; options: any } | null = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string, options: any) => {
      lastFetchCall = { url, options };
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            choices: [{ message: { content: "OK" } }],
          }),
      };
    }) as any;

    try {
      await callAIProvider({
        provider: "deepseek",
        apiKey: "dummy-key",
        model: "deepseek-chat",
        systemInstruction: "You are a helpful assistant.",
        prompt: "Hello world",
      });

      expect(lastFetchCall).toBeTruthy();
      const chatBody = JSON.parse(lastFetchCall!.options.body);
      expect(chatBody.model).toBe("deepseek-chat");
      expect(chatBody.messages).toEqual([
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello world" },
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("deepseek-reasoner model (systemInstruction merged into user prompt)", async () => {
    let lastFetchCall: { url: string; options: any } | null = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string, options: any) => {
      lastFetchCall = { url, options };
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            choices: [{ message: { content: "OK" } }],
          }),
      };
    }) as any;

    try {
      await callAIProvider({
        provider: "deepseek",
        apiKey: "dummy-key",
        model: "deepseek-reasoner",
        systemInstruction: "You are a reasoning assistant.",
        prompt: "Solve this math problem",
      });

      expect(lastFetchCall).toBeTruthy();
      const reasonerBody = JSON.parse(lastFetchCall!.options.body);
      expect(reasonerBody.model).toBe("deepseek-reasoner");
      expect(reasonerBody.messages).toEqual([
        {
          role: "user",
          content: "You are a reasoning assistant.\n\nSolve this math problem",
        },
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("deepseek-reasoner without systemInstruction", async () => {
    let lastFetchCall: { url: string; options: any } | null = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string, options: any) => {
      lastFetchCall = { url, options };
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            choices: [{ message: { content: "OK" } }],
          }),
      };
    }) as any;

    try {
      await callAIProvider({
        provider: "deepseek",
        apiKey: "dummy-key",
        model: "deepseek-reasoner",
        systemInstruction: "",
        prompt: "Solve this without system instruction",
      });

      expect(lastFetchCall).toBeTruthy();
      const reasonerBodyNoSys = JSON.parse(lastFetchCall!.options.body);
      expect(reasonerBodyNoSys.model).toBe("deepseek-reasoner");
      expect(reasonerBodyNoSys.messages).toEqual([
        {
          role: "user",
          content: "Solve this without system instruction",
        },
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
