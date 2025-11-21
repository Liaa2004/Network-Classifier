const { spawn } = require("child_process");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");
const { v4: uuidv4 } = require("uuid");

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:5000";

module.exports = {
  
  analyzePcapFile: async function (filePath, uploadId) {
    return new Promise((resolve, reject) => {

      const csvOutput = path.join(
        path.dirname(filePath),
        `${uploadId}_flows.csv`
      );

      // Step 1: Run pcap2csv to convert PCAP -> CSV
      const py = spawn("python", [
        path.join(__dirname, "..", "python", "pcap2csv_win_new.py"),
        filePath,
        csvOutput
      ]);

      py.stdout.on("data", (d) => console.log("[PCAP2CSV]", d.toString()));
      py.stderr.on("data", (d) => console.error("[PCAP2CSV ERROR]", d.toString()));

      py.on("close", async (code) => {
        if (code !== 0) {
          return reject(new Error("pcap2csv conversion failed"));
        }

        console.log("[DONE] CSV created:", csvOutput);

        // Step 2: Read CSV -> Extract flows
        const flows = [];

        fs.createReadStream(csvOutput)
          .pipe(csv())
          .on("data", (row) => {
            const flow = {
              row_id: uuidv4(),
              upload_id: uploadId,
              ...row
            };
            flows.push(flow);
          })
          .on("end", async () => {
            console.log(`[INFO] Extracted ${flows.length} flows`);

            // Step 3: Send flows to FastAPI ML server
            let sentCount = 0;

            for (const flow of flows) {
              try {
                await axios.post(`${FASTAPI_URL}/api/flows`, flow);
                sentCount++;
              } catch (err) {
                console.error("[FASTAPI ERROR]", err.message);
              }
            }

            console.log(`[INFO] Sent ${sentCount}/${flows.length} flows to ML server`);

            resolve({
              uploadId,
              totalFlows: flows.length,
              sentToML: sentCount,
              csvOutput
            });
          });
      });
    });
  }
};
