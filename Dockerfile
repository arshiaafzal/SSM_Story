FROM ruby:3.1-slim

# Install dependencies for Jekyll & bundler
RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  nodejs \
  npm \
  && rm -rf /var/lib/apt/lists/*

# Install bundler & Jekyll
RUN gem install bundler jekyll

WORKDIR /srv/jekyll
COPY . /srv/jekyll

# Install project gems
RUN bundle install

EXPOSE 4000
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload", "--force_polling"]
