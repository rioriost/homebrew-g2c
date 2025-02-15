class G2c < Formula
  include Language::Python::Virtualenv

  desc "g2c is a python script to convert Gremlin query to Cypher query with OpenAI API"
  homepage "https://github.com/rioriost/homebrew-g2c/"
  url "https://files.pythonhosted.org/packages/c5/17/5f46ae790991d822a17eeef3cdcf999bef0d5113ec3663a225b5e45fd4d7/g2c-0.1.0.tar.gz"
  sha256 "76e53fd3113f406e9de81ba8876fcd43cc0342869f83559cd902b708a2ee8ae1"
  license "MIT"

  depends_on "python@3.11"

  resource "openai" do
    url "https://files.pythonhosted.org/packages/4f/32/2049e973a646801df425aecdf88c6504ca878bdb3951fe12076fc30f2977/openai-1.63.0.tar.gz"
    sha256 "597d7a1b35b113e5a09fcb953bdb1eef44f404a39985f3d7573b3ab09221fd66"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/b7/ae/d5220c5c52b158b1de7ca89fc5edb72f304a70a4c540c84c8844bf4008de/pydantic-2.10.6.tar.gz"
    sha256 "ca5daa827cce33de7a42be142548b0096bf05a7e7b365aebfa5f8eeec7128236"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{bin}/g2c", "--help"
  end
end
