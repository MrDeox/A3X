class A3XError(Exception):
    """Base class for custom exceptions in the A3X project."""
    pass

class OrchestrationError(A3XError):
    """Exception raised for errors during task orchestration."""
    pass

class ToolExecutionError(A3XError):
    """Exception raised when a tool or skill fails to execute."""
    pass

class ConfigurationError(A3XError):
    """Exception raised for configuration-related errors."""
    pass

class LLMCommunicationError(A3XError):
    """Exception raised for errors communicating with the LLM."""
    pass

class SkillNotFoundError(A3XError):
    """Exception raised when a requested skill is not found in the registry."""
    pass

class SkillRegistrationError(A3XError):
    """Exception raised during the registration of a skill."""
    pass

class FragmentExecutionError(A3XError):
    """Exception raised when a Fragment fails during execution."""
    def __init__(self, message, status="error", reason="fragment_execution_failed"):
        super().__init__(message)
        self.status = status
        self.reason = reason 