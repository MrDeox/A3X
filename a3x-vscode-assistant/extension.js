const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {

    console.log('Congratulations, your extension "a3x-assistant" is now active!');

    let disposable = vscode.commands.registerCommand('a3x.runTask', async function () {
        // Get the root path of the current workspace
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            vscode.window.showErrorMessage('Nenhum workspace aberto. Abra o projeto A³X.');
            return;
        }
        const workspaceRoot = workspaceFolders[0].uri.fsPath;

        // Get the path to the python executable in the virtual environment
        // TODO: Make this configurable
        const pythonExecutable = path.join(workspaceRoot, '.venv', 'bin', 'python'); // Assuming standard venv location
        const scriptPath = path.join(workspaceRoot, 'a3x', 'assistant_cli.py');

        // Prompt user for the task
        const taskDescription = await vscode.window.showInputBox({ 
            prompt: "Descreva a task para o A³X Agent",
            placeHolder: "Ex: Analise este arquivo e extraia as funções úteis"
        });

        if (!taskDescription) {
            vscode.window.showInformationMessage('Nenhuma task fornecida.');
            return;
        }

        // Show progress notification
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "A³X está pensando...",
            cancellable: false // TODO: Consider making this cancellable
        }, async (progress) => {

            progress.report({ increment: 0 });

            // Construct the command
            // Ensure taskDescription is properly escaped for the shell
            const escapedTaskDescription = taskDescription.replace(/"/g, '\\"');
            const command = `${pythonExecutable} ${scriptPath} --task "${escapedTaskDescription}"`;

            // Execute the command
            return new Promise((resolve, reject) => {
                // Increase maxBuffer size if needed (e.g., for large outputs)
                const child = exec(command, { cwd: workspaceRoot, maxBuffer: 1024 * 1024 }, (error, stdout, stderr) => {
                    if (error) {
                        console.error(`exec error: ${error}`);
                        vscode.window.showErrorMessage(`Erro ao executar A³X: ${stderr || error.message}`);
                        reject(error);
                        return;
                    }

                    // Assuming the final answer is the last non-empty line of stdout
                    const lines = stdout.trim().split('\n');
                    const finalAnswer = lines[lines.length - 1]; // Adjust if output format differs

                    vscode.window.showInformationMessage(`A³X Respondeu: ${finalAnswer}`);
                    console.log(`stdout: ${stdout}`);
                    if (stderr) {
                        console.error(`stderr: ${stderr}`);
                    }
                    resolve();
                });
            });
        });
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
} 