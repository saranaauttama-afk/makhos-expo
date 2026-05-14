const data = require('../.tmp/benchmarks/ai-benchmark-quick-latest.json');

for (const level of ['easy', 'normal', 'hard', 'expert']) {
  const fails = data.tacticalSamples.filter(s => s.level === level && (!s.solved || s.severeBlunder));
  if (fails.length > 0) {
    console.log(`\n${level} failures:`);
    fails.forEach(f => console.log(`  ${f.caseId}: chose ${f.chosenMove}, oracle ${f.oracleMove}, drop=${f.scoreDrop}`));
  } else {
    console.log(`\n${level}: ✅ No failures!`);
  }
}
