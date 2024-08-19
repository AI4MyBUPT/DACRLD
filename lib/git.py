from git import Repo
import glob

def git_commit(
    work_dir,
    timestamp,
    levels=5,
    postfixs=[".py", ".sh", ".yaml"],
    commit_info="",
    debug=False,
):
    cid = "not generate"
    branch = "unknown"
    if not debug:
        repo = Repo(work_dir)
        toadd = []
        branch = repo.active_branch.name
        for i in range(levels):
            for postfix in postfixs:
                filename = glob.glob(work_dir + (i + 1) * "/*" + postfix)
                for x in filename:
                    if (
                        not ("play" in x)
                        and not ("local" in x)
                        and not ("Untitled" in x)
                        and not ("wandb" in x)
                        and not ("output" in x)
                    ):
                        # check for gitignored files
                        if repo.ignored(x):
                            continue
                        # delete the ./ prefix in the filename
                        if x[0:2] == "./":
                            x = x[2:]
                        toadd.append(x)
        index = repo.index  # 获取暂存区对象
        index.add(toadd)
        index.commit(commit_info)
        cid = repo.head.commit.hexsha

    commit_tag = (
        commit_info
        + "\n"
        + "COMMIT BRANCH >>> "
        + branch
        + " <<< \n"
        + "COMMIT ID >>> "
        + cid
        + " <<<"
    )
    record_commit_info = " COMMIT TAG [ \n %s ]\n" % commit_tag
    return record_commit_info