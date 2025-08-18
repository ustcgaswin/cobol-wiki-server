#!/usr/bin/env node
// mmd-check.js
// Usage: echo "<mermaid>" | node mmd-check.js
// Exit 0  -> valid
// Exit 1  -> invalid
// On invalid, prints the 1-based line number of the first syntax error
// followed by the raw error message.

const fs = require('fs');
require('jsdom-global')();          // stub browser globals

// Use an async IIFE (Immediately Invoked Function Expression) to handle the dynamic import
(async () => {
  try {
    // Dynamically import the mermaid ESM module. The actual library is the default export.
    const { default: mermaid } = await import('mermaid');
    
    const src = fs.readFileSync(0, 'utf8');

    // Initialize the API before parsing. This is good practice with recent versions.
    mermaid.initialize({ startOnLoad: false });

    // The parse method now returns a promise and a boolean, so we await it.
    await mermaid.parse(src);
    
    // If parse() does not throw an error, the syntax is valid.
    process.exit(0);                // OK

  } catch (e) {
    // Extract line number from mermaid error string
    const match = e.message.match(/line:? (\d+)/i); // Make regex more flexible
    const lineNum = match ? parseInt(match[1], 10) : 0;
    console.log(lineNum);             // first line of output
    console.error(e.message);       // second line (stderr)
    process.exit(1);                // NOT OK
  }
})();