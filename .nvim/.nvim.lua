-- ~/.my-project/.nvim.lua

print("Loading project-specific conform.nvim config...")

local status_ok, conform = pcall(require, "conform")

if not status_ok then
	print("conform.nvim not found, skipping project config.")
	return
end

conform.setup({
	formatters = {
		ruff_format = { command = ".venv/bin/ruff" },
		ruff_organize_imports = { command = ".venv/bin/ruff" },
	},
	-- This table will be merged with your global formatters_by_ft
	formatters_by_ft = {
		-- Maybe use a different formatter for python in this project
		python = { "ruff_organize_imports", "ruff_format" }, -- Assuming ruff_format is defined globally or below

		-- You can also add formatters for filetypes not configured globally
		-- typescript = { "prettier" },
	},
})

print("Project-specific conform.nvim config applied.")

-- You might want to add a command or mapping to easily check the config
vim.api.nvim_create_user_command("ProjectConformInfo", function()
	require("conform").info()
end, {})
