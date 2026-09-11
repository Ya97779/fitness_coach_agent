"""LLM provider configuration tests."""

import os
import sys
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.llm_manager import GLMChatOpenAI, LLMManager


class TestLLMManager(unittest.TestCase):
    def tearDown(self):
        LLMManager.clear()

    @patch("app.llm_manager.GLMChatOpenAI")
    def test_glm_unsupported_medium_reasoning_falls_back_to_low(self, mock_chat_openai):
        with patch.dict(os.environ, {
            "LLM_MODEL": "glm-5.3-flash",
            "LLM_REASONING_EFFORT": "medium",
            "LLM_THINKING_TYPE": "enabled",
            "LLM_CLEAR_THINKING": "false",
        }):
            LLMManager.clear()
            LLMManager.get_llm(temperature=0.1)

        kwargs = mock_chat_openai.call_args.kwargs
        self.assertEqual(kwargs["model"], "glm-5.3-flash")
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["extra_body"], {
            "thinking": {
                "type": "enabled",
                "clear_thinking": False,
            }
        })

    def test_glm_stream_preserves_reasoning_content(self):
        model = GLMChatOpenAI.model_construct()
        converted = ChatGenerationChunk(
            message=AIMessageChunk(content="")
        )
        raw_chunk = {
            "choices": [{"delta": {"reasoning_content": "step"}}]
        }

        with patch.object(
            ChatOpenAI,
            "_convert_chunk_to_generation_chunk",
            return_value=converted,
        ):
            result = model._convert_chunk_to_generation_chunk(
                raw_chunk, AIMessageChunk, None
            )

        self.assertEqual(
            result.message.additional_kwargs["reasoning_content"],
            "step",
        )


if __name__ == "__main__":
    unittest.main()
