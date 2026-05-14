const data = require('../.tmp/benchmarks/ai-benchmark-quick-latest.json');
const expert = data.tacticalSummary.find(s => s.level === 'expert');
console.log('Expert:', Math.round(expert.solveRate*100)+'% solve,', Math.round(expert.blunderRate*100)+'% blunder');
const failures = data.tacticalSamples.filter(s => s.level === 'expert' && (!s.solved || s.severeBlunder));
console.log('\nFailures:');
failures.forEach(f => console.log('  '+f.caseId+':', 'chose', f.chosenMove, 'oracle', f.oracleMove, 'drop='+f.scoreDrop));
