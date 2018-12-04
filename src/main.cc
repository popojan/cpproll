
#include <iostream>
#include <thread>
#include <clipp.h>
#include <csignal>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "stage1.h"
#include "stage2.h"
#include "stage3.h"
#include "options.h"



volatile bool interrupted = false;

void signalHandler(int signum) {
    interrupted = true;
    std::cout << std::endl;
    spdlog::get("console")->info("User termination in progress.");
}

int main(int argc, char* argv[]) {

    std::locale::global(std::locale("C"));

    Options opts;

    using namespace clipp;

    auto cli = (
        (option("-b", "--bits") & integer("bits", opts.B))
            .doc("size of feature indices created by the hashing trick"),
        (option("-B", "--batch") & integer("size", opts.batch))
            .doc("number of examples in each training batch (experimental)"),
        (option("--seed") & integer("seed", opts.seed))
            .doc("murmur3 hash seed used for the hashing trick"),
        (option("--predict_sample") & integer("size", opts.npredict))
            .doc("number of extra sampled predictions to be made on each example"),
        (option("-v", "--verbose") & word("level", opts.verbose))
            .doc("output verbosity: trace, debug, info, warning, error, critical, off"),
        (option("-D", "--description") & value("regexp", opts.desc))
            .doc("regular expression selecting features used to describe predictions"),
        (option("-s", "--standardize").set(opts.standardize))
            .doc("running standardization to mean 1 var 1 on non-zero features"),
        (option("-t", "--test_only").set(opts.testonly))
            .doc("do not train at all on examples (neither after predicting)"),
        (option("-e", "--explain").set(opts.explain))
            .doc("multiline predictions explaining contribution of every feature"),
        (option("-p", "--predictions") & value("file", opts.fpred))
            .doc("file to write predictions into"),
        (option("-i", "--initial_regressor") & value("prefix", opts.fmodelin))
            .doc("initialize the model from previously saved model"),
        (option("-f", "--final_regressor") & value("prefix", opts.fmodel))
            .doc("name of the final model to be saved"),
        (option("--logit") & value("regexp", opts.relogit))
            .doc("regular expression selecting primary features for logit transform"),
        (option("--log") & value("regexp", opts.relog))
            .doc("regular expression selecting primary features for log transform"),
        (option("--svmlight") & value("file", opts.svmlight))
            .doc("extra file to output feature-hashed data into"),
        (option("--ignore") & value("regexp", opts.reignore))
            .doc("regular expression selecting features to be ignored"),
        (option("--keep") & value("regexp", opts.rekeep))
            .doc("regular expression selecting features to be kept"),
        (option("-I", "--interactions") & value("a*b*c,x*y", opts.interactions))
            .doc("namespace interactions to be added as additional features"),
        (option("-j", "--jobs") & integer("num", opts.jobs))
            .doc("number of parallel threads processing the data"),
        (option("--l2")  & number("regularization", opts.lambda))
            .doc("regularization lambda (initial precision of the parameters)"),
        (option("--passes") & integer("num", opts.passes))
            .doc("number of passes over the data (uderestimates variance)"),
        (option("-T", "--T") & integer("ms", opts.period))
            .doc("period in milliseconds to regularly output current metrics"),
        values("file", opts.files)
    );

    if(parse(argc, argv, cli)) {
        auto console = spdlog::stdout_color_mt("console");

        spdlog::set_level(spdlog::level::from_str(opts.verbose));

        console->info("Welcome to Cuddly T-Rex!");
        console->debug("L-BFGS precision [{0}]", LBFGS_FLOAT);
        console->debug("Setting n_jobs = {0}", opts.jobs);

        console->debug("Input data file [{0}]", opts.files.size());

        opts.N = 1ul << opts.B;
        console->info("Feature table size 2^{0} = {1}", opts.B, opts.N);

        Buffer<std::string> b;
        Buffer<Batch> b2;

        Stage1<std::string> p(b, opts);
        Stage2<std::string, Batch> c(b, b2, opts);
        Stage3<Batch> c2(b2, opts);

        if(!opts.fmodelin.empty()) {
            console->info("Loading model from [{0}*]", opts.fmodelin);
            c2.load(opts.fmodelin);
        }

        std::signal(SIGINT, signalHandler);

        std::thread stage1(&Stage1<std::string>::run, &p);

        std::vector<std::thread> stage2;
        for(int i = 0; i < opts.jobs; ++i) {
            stage2.push_back(
                std::thread(&Stage2<std::string, Batch>::run, &c)
            );
        }
        std::thread stage3(&Stage3<Batch>::run, &c2);

        stage1.join();
        for(int i = 0; i < opts.jobs; ++i) {
            stage2[i].join();
        }
        stage3.join();

        spdlog::set_level(spdlog::level::info);
        c2.eval();
        spdlog::set_level(spdlog::level::from_str(opts.verbose));

        if(!opts.fmodel.empty()) {
            console->info("Saving model to [{0}*]", opts.fmodel);
            c2.save(opts.fmodel);
        }

    } else {
        auto fmt = doc_formatting{}.doc_column(12);
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';
    }

    return 0;
}
